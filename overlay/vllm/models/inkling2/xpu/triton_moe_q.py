# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantised-expert kernels for Inkling2 on XPU (decode + dequant for prefill).

Formats (match scripts/inkling2_dequant.py, verified against llama.cpp):
  NVFP4 : packed uint8 [E, R, K/2] (low nibble = even k), e4m3 block scales
          [E, R, K/16], per-expert outer scale = amax / denominator.
  MXFP8 : e4m3 bytes [E, R, K], ue8m0 scales [E, R, K/32] (2 ** (b - 127)).
Rows of w13 are stored in packed [gates; ups] order (permuted at load).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

FMT_NVFP4 = 1
FMT_MXFP8 = 2


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _e2m1(nib):
    sign = (nib >> 3) & 1
    m = nib & 7
    e = m >> 1
    b = (m & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * b, (1.0 + 0.5 * b) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _e2m1_fast(nib):
    """Bit-exact twin of `_e2m1` built from integer ops + bitcast (no exp2 / float select chain)."""
    nib = nib.to(tl.int32)
    m = nib & 7
    e = m >> 1
    bits_n = ((e + 126) << 23) | ((m & 1) << 22)
    bits_s = tl.where((m & 1) == 1, 0x3F000000, 0)
    bits = tl.where(e > 0, bits_n, bits_s) | ((nib & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _e4m3_fast(byte):
    """Bit-exact twin of `_e4m3` (subnormals via an exact int->float multiply, normals via bitcast)."""
    byte = byte.to(tl.int32)
    e = (byte >> 3) & 15
    m = byte & 7
    val_n = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    val = tl.where(e == 0, m.to(tl.float32) * 0.001953125, val_n)
    return tl.where((byte & 0x80) != 0, -val, val)


@triton.jit
def _e4m3(byte):
    s = (byte >> 7) & 1
    e = (byte >> 3) & 15
    m = (byte & 7).to(tl.float32)
    val = tl.where(e == 0, m * 0.125 * 0.015625, (1.0 + m * 0.125) * tl.exp2((e - 7).to(tl.float32)))
    return tl.where(s == 1, -val, val)


@triton.jit
def _qdot(
    W,  # base pointer of this expert's [R, K*] weight matrix
    S,  # base pointer of this expert's scales
    X,  # fp32/bf16 vector pointer (length K)
    rows,  # [BN] row ids
    rmask,
    K,
    stride_wn,
    stride_sn,
    BD: tl.constexpr,
    FMT: tl.constexpr,
):
    """sum_k dequant(W[rows, k]) * X[k] (without the per-expert outer scale)."""
    acc = tl.zeros([rows.shape[0]], tl.float32)
    if FMT == 1:
        jj = tl.arange(0, BD // 2)
        for d0 in range(0, K, BD):
            pk = tl.load(
                W + rows[:, None] * stride_wn + (d0 // 2 + jj)[None, :], mask=rmask[:, None], other=0
            ).to(tl.int32)
            sc = tl.load(
                S + rows[:, None] * stride_sn + (d0 // 16 + jj // 8)[None, :], mask=rmask[:, None], other=0
            ).to(tl.int32)
            scale = _e4m3(sc)
            xe = tl.load(X + d0 + 2 * jj).to(tl.float32)
            xo = tl.load(X + d0 + 2 * jj + 1).to(tl.float32)
            v = _e2m1(pk & 15) * xe[None, :] + _e2m1((pk >> 4) & 15) * xo[None, :]
            acc += tl.sum(v * scale, 1)
    else:
        kk = tl.arange(0, BD)
        for d0 in range(0, K, BD):
            pk = tl.load(W + rows[:, None] * stride_wn + (d0 + kk)[None, :], mask=rmask[:, None], other=0).to(
                tl.int32
            )
            sc = tl.load(
                S + rows[:, None] * stride_sn + ((d0 + kk) // 32)[None, :], mask=rmask[:, None], other=127
            ).to(tl.int32)
            scale = tl.exp2((sc - 127).to(tl.float32))
            x = tl.load(X + d0 + kk).to(tl.float32)
            acc += tl.sum(_e4m3(pk) * scale * x[None, :], 1)
    return acc


@triton.jit
def _q_gate_up_kernel(
    X,
    W13,
    S13,
    Outer,
    Idx,
    Act,
    top_k,
    ep_start,
    num_local,
    hidden,
    inter,
    stride_xt,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    beta,
    linear_beta,
    BN: tl.constexpr,
    BD: tl.constexpr,
    FMT: tl.constexpr,
):
    p = tl.program_id(0)
    nb = tl.program_id(1)
    t = p // top_k
    e = tl.load(Idx + p) - ep_start
    n = nb * BN + tl.arange(0, BN)
    nmask = n < inter
    valid = (e >= 0) & (e < num_local)
    e = tl.maximum(e, 0).to(tl.int64)
    acc_g = tl.zeros([BN], tl.float32)
    acc_u = tl.zeros([BN], tl.float32)
    if valid:
        outer = tl.load(Outer + e)
        acc_g = _qdot(W13 + e * stride_we, S13 + e * stride_se, X + t * stride_xt, n, nmask, hidden,
                      stride_wn, stride_sn, BD, FMT) * outer
        acc_u = _qdot(W13 + e * stride_we, S13 + e * stride_se, X + t * stride_xt, inter + n, nmask, hidden,
                      stride_wn, stride_sn, BD, FMT) * outer
    g = beta * _tanh(acc_g / beta) * (1.0 / (1.0 + tl.exp(-acc_g)))
    u = linear_beta * _tanh(acc_u / linear_beta)
    tl.store(Act + p * inter + n, (g * u).to(Act.dtype.element_ty), mask=nmask)


@triton.jit
def _q_down_kernel(
    Act,
    W2,
    S2,
    Outer,
    Idx,
    Wts,
    Y,
    ep_start,
    num_local,
    hidden,
    inter,
    stride_we,
    stride_wh,
    stride_se,
    stride_sh,
    BH: tl.constexpr,
    BI: tl.constexpr,
    FMT: tl.constexpr,
):
    p = tl.program_id(0)
    hb = tl.program_id(1)
    e = tl.load(Idx + p) - ep_start
    w = tl.load(Wts + p)
    h = hb * BH + tl.arange(0, BH)
    hm = h < hidden
    valid = (e >= 0) & (e < num_local)
    e = tl.maximum(e, 0).to(tl.int64)
    acc = tl.zeros([BH], tl.float32)
    if valid:
        outer = tl.load(Outer + e)
        acc = _qdot(W2 + e * stride_we, S2 + e * stride_se, Act + p * inter, h, hm, inter,
                    stride_wh, stride_sh, BI, FMT) * outer
    y = acc.to(tl.bfloat16).to(tl.float32) * w
    y = tl.where(valid, y, 0.0)
    tl.store(Y + p * hidden + h, y, mask=hm)


@triton.jit
def _nib16(t):
    """fp16 bits (s<<15 | eem<<9) of the nibble pre-shifted to bits 15:12 -> f32 = value * 2**-14."""
    return ((t & 0x8000) | ((t >> 3) & 0x0E00)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _qdot16(W, S, X, rows, k0, stride_wn, stride_sn, BD: tl.constexpr, FMT: tl.constexpr):
    """sum over one BD-wide k chunk starting at k0 of dequant(W[rows, k]) * X[k].

    Weights are read as uint16 groups (4 fp4 or 2 fp8 elements per group) and decoded
    to fp32 through exact fp16 bit patterns (llama.cpp mul_mat_vec_nvfp4 idea):
      fp4 nibble s.eem  -> f16 bits s<<15|eem<<9 = value * 2**-14
      fp8 byte  s.eeeemm-> f16 bits (b&0x7F)<<7|sign<<15 = value * 2**-8
    Block scales are folded so the result equals sum dequant(w)*x up to fp32
    rounding order (the group-level sum is factored out of the block scale).
    """
    if FMT == 1:
        G: tl.constexpr = BD // 4
        gg = tl.arange(0, G)
        pk = tl.load(W + rows[:, None] * stride_wn + (k0 // 4 + gg)[None, :]).to(tl.int32) & 0xFFFF
        v0 = _nib16(pk << 12)
        v1 = _nib16(pk << 8)
        v2 = _nib16(pk << 4)
        v3 = _nib16(pk)
        x0 = tl.load(X + k0 + 4 * gg).to(tl.float32)
        x1 = tl.load(X + k0 + 4 * gg + 1).to(tl.float32)
        x2 = tl.load(X + k0 + 4 * gg + 2).to(tl.float32)
        x3 = tl.load(X + k0 + 4 * gg + 3).to(tl.float32)
        pr = v0 * x0[None, :] + v1 * x1[None, :] + v2 * x2[None, :] + v3 * x3[None, :]
        blk = tl.sum(tl.reshape(pr, (rows.shape[0], G // 4, 4)), 2)
        sb = tl.load(S + rows[:, None] * stride_sn + (k0 // 16 + tl.arange(0, G // 4))[None, :]).to(tl.int32)
        sfb = (((sb & 0x7F) << 7) | ((sb & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        return tl.sum(blk * (sfb * (2.0**22)), 1)
    else:
        G: tl.constexpr = BD // 2
        gg = tl.arange(0, G)
        pk = tl.load(W + rows[:, None] * stride_wn + (k0 // 2 + gg)[None, :]).to(tl.int32) & 0xFFFF
        blo, bhi = pk & 0xFF, (pk >> 8) & 0xFF
        vlo = (((blo & 0x7F) << 7) | ((blo & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        vhi = (((bhi & 0x7F) << 7) | ((bhi & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        xe = tl.load(X + k0 + 2 * gg).to(tl.float32)
        xo = tl.load(X + k0 + 2 * gg + 1).to(tl.float32)
        pr = vlo * xe[None, :] + vhi * xo[None, :]
        blk = tl.sum(tl.reshape(pr, (rows.shape[0], G // 16, 16)), 2)
        sb = tl.load(S + rows[:, None] * stride_sn + (k0 // 32 + tl.arange(0, G // 16))[None, :]).to(tl.int32)
        sp = (sb << 23).to(tl.float32, bitcast=True) * 256.0
        return tl.sum(blk * sp, 1)


@triton.jit
def _qg_split_kernel(
    X, W13, S13, Outer, Idx, WS, Act,
    top_k, ep_start, num_local, inter,
    stride_xt, stride_we, stride_wn, stride_se, stride_sn,
    beta, linear_beta,
    NBG: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    KD: tl.constexpr, SKG: tl.constexpr, FMT: tl.constexpr,
):
    p = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.program_id(2)
    t = p // top_k
    e = tl.load(Idx + p) - ep_start
    n = nb * BN + tl.arange(0, BN)
    valid = (e >= 0) & (e < num_local)
    e = tl.maximum(e, 0).to(tl.int64)
    KCH: tl.constexpr = KD // SKG
    k0 = s * KCH
    acc_g = tl.zeros([BN], tl.float32)
    acc_u = tl.zeros([BN], tl.float32)
    if valid:
        wbase = W13 + e * stride_we
        sbase = S13 + e * stride_se
        for i in tl.static_range(KCH // BD):
            d0 = k0 + i * BD
            acc_g += _qdot16(wbase, sbase, X + t * stride_xt, n, d0, stride_wn, stride_sn, BD, FMT)
            acc_u += _qdot16(wbase, sbase, X + t * stride_xt, inter + n, d0, stride_wn, stride_sn, BD, FMT)
    if SKG == 1:
        outer = tl.load(Outer + e)
        g = beta * _tanh(acc_g * outer / beta) * (1.0 / (1.0 + tl.exp(-acc_g * outer)))
        u = linear_beta * _tanh(acc_u * outer / linear_beta)
        tl.store(Act + p * inter + n, (g * u).to(Act.dtype.element_ty))
    else:
        off = (p * NBG + nb) * SKG * 2 * BN + s * 2 * BN
        jj = tl.arange(0, BN)
        tl.store(WS + off + jj, acc_g)
        tl.store(WS + off + BN + jj, acc_u)


@triton.jit
def _qg_combine_kernel(
    WS, Outer, Idx, Act, ep_start, inter, beta, linear_beta,
    NBG: tl.constexpr, BN: tl.constexpr, SKG: tl.constexpr, SKP: tl.constexpr,
):
    pp = tl.program_id(0)
    p = pp // NBG
    nb = pp % NBG
    n = nb * BN + tl.arange(0, BN)
    ss = tl.arange(0, SKP)
    sm = ss < SKG
    jj = tl.arange(0, BN)
    ag = tl.sum(tl.load(WS + (pp * SKG + ss[:, None]) * 2 * BN + jj[None, :], mask=sm[:, None], other=0.0), 0)
    au = tl.sum(tl.load(WS + (pp * SKG + ss[:, None]) * 2 * BN + BN + jj[None, :], mask=sm[:, None], other=0.0), 0)
    e = tl.load(Idx + p) - ep_start
    outer = tl.load(Outer + tl.maximum(e, 0))
    g = beta * _tanh(ag * outer / beta) * (1.0 / (1.0 + tl.exp(-ag * outer)))
    u = linear_beta * _tanh(au * outer / linear_beta)
    tl.store(Act + p * inter + n, (g * u).to(Act.dtype.element_ty))


@triton.jit
def _qd_split_kernel(
    Act, W2, S2, Outer, Idx, Wts, WS, Y,
    ep_start, num_local, hidden, inter,
    stride_we, stride_wh, stride_se, stride_sh,
    NH: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
    KI: tl.constexpr, SKD: tl.constexpr, FMT: tl.constexpr,
):
    p = tl.program_id(0)
    hb = tl.program_id(1)
    s = tl.program_id(2)
    e = tl.load(Idx + p) - ep_start
    h = hb * BH + tl.arange(0, BH)
    valid = (e >= 0) & (e < num_local)
    e = tl.maximum(e, 0).to(tl.int64)
    KCH: tl.constexpr = KI // SKD
    k0 = s * KCH
    acc = tl.zeros([BH], tl.float32)
    if valid:
        for i in tl.static_range(KCH // BD):
            d0 = k0 + i * BD
            acc += _qdot16(W2 + e * stride_we, S2 + e * stride_se, Act + p * inter, h, d0,
                           stride_wh, stride_sh, BD, FMT)
    if SKD == 1:
        outer = tl.load(Outer + e)
        w = tl.load(Wts + p)
        y = (acc * outer).to(tl.bfloat16).to(tl.float32) * w
        tl.store(Y + p * hidden + h, y)
    else:
        off = (p * NH + hb) * SKD * BH + s * BH
        tl.store(WS + off + tl.arange(0, BH), acc)


@triton.jit
def _qd_combine_kernel(
    WS, Outer, Idx, Wts, Y, ep_start, hidden,
    NH: tl.constexpr, BH: tl.constexpr, SKD: tl.constexpr, SKP: tl.constexpr,
):
    pp = tl.program_id(0)
    p = pp // NH
    hb = pp % NH
    h = hb * BH + tl.arange(0, BH)
    ss = tl.arange(0, SKP)
    acc = tl.sum(tl.load(WS + (pp * SKD + ss[:, None]) * BH + tl.arange(0, BH)[None, :],
                         mask=(ss < SKD)[:, None], other=0.0), 0)
    e = tl.load(Idx + p) - ep_start
    outer = tl.load(Outer + tl.maximum(e, 0))
    w = tl.load(Wts + p)
    y = (acc * outer).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + p * hidden + h, y)


_SPLIT_DEFAULT = (9, 32, 256, 4, 1, 16, 128, 4)  # skg,bn,bd,nwg, skd,bh,bd2,nwd


def _split_cfg():
    v = os.environ.get("INKLING_MOEQ_SPLIT")
    if not v:
        return _SPLIT_DEFAULT
    t = tuple(int(x) for x in v.split(","))
    return t if len(t) == 8 else _SPLIT_DEFAULT


@torch.no_grad()
def _moe_small_q_split(x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt):
    """Split-K decode path; returns None if the shapes/config do not fit it."""
    if fmt not in (FMT_NVFP4, FMT_MXFP8) or not (x.is_contiguous() and w13.is_contiguous() and w2.is_contiguous()):
        return None
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    pairs = num_tokens * top_k
    skg, bn, bd, nwg, skd, bh, bd2, nwd = _split_cfg()
    if hidden % (skg * bd) or inter % (skd * bd2) or inter % bn or hidden % bh or pairs == 0:
        return None
    idx = indices.reshape(-1).to(torch.int32).contiguous()
    wts = weights.reshape(-1).float().contiguous()
    x = x.contiguous()
    w13v = w13.view(torch.int16)
    w2v = w2.view(torch.int16)
    nbg = inter // bn
    nh = hidden // bh
    act = torch.empty(pairs, inter, device=x.device, dtype=torch.bfloat16)
    y = torch.empty(pairs, hidden, device=x.device, dtype=torch.float32)
    gu_args = (x, w13v, s13, o13, idx, act, act, top_k, ep_start, num_local, inter,
               x.stride(0), w13v.stride(0), w13v.stride(1), s13.stride(0), s13.stride(1),
               beta, linear_beta)
    if skg == 1:
        _qg_split_kernel[(pairs, nbg, 1)](*gu_args, NBG=nbg, BN=bn, BD=bd, KD=hidden,
                                          SKG=1, FMT=fmt, num_warps=nwg)
    else:
        ws = torch.empty(pairs * nbg, skg, 2, bn, device=x.device, dtype=torch.float32)
        _qg_split_kernel[(pairs, nbg, skg)](x, w13v, s13, o13, idx, ws, act, top_k, ep_start,
                                            num_local, inter,
                                            x.stride(0), w13v.stride(0), w13v.stride(1),
                                            s13.stride(0), s13.stride(1), beta, linear_beta,
                                            NBG=nbg, BN=bn, BD=bd, KD=hidden, SKG=skg, FMT=fmt,
                                            num_warps=nwg)
        _qg_combine_kernel[(pairs * nbg,)](ws, o13, idx, act, ep_start, inter, beta, linear_beta,
                                           NBG=nbg, BN=bn, SKG=skg, SKP=1 << (skg - 1).bit_length(),
                                           num_warps=1)
    d_args = (act, w2v, s2, o2, idx, wts, y, y, ep_start, num_local, hidden, inter,
              w2v.stride(0), w2v.stride(1), s2.stride(0), s2.stride(1))
    if skd == 1:
        _qd_split_kernel[(pairs, nh, 1)](*d_args, NH=nh, BH=bh, BD=bd2, KI=inter, SKD=1,
                                         FMT=fmt, num_warps=nwd)
    else:
        wsd = torch.empty(pairs * nh, skd, bh, device=x.device, dtype=torch.float32)
        _qd_split_kernel[(pairs, nh, skd)](act, w2v, s2, o2, idx, wts, wsd, y, ep_start,
                                           num_local, hidden, inter,
                                           w2v.stride(0), w2v.stride(1), s2.stride(0), s2.stride(1),
                                           NH=nh, BH=bh, BD=bd2, KI=inter, SKD=skd, FMT=fmt,
                                           num_warps=nwd)
        _qd_combine_kernel[(pairs * nh,)](wsd, o2, idx, wts, y, ep_start, hidden,
                                          NH=nh, BH=bh, SKD=skd, SKP=1 << (skd - 1).bit_length(),
                                          num_warps=1)
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)


@torch.no_grad()
def _moe_small_q_base(
    x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt,
):
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    pairs = num_tokens * top_k
    idx = indices.reshape(-1).to(torch.int32).contiguous()
    wts = weights.reshape(-1).float().contiguous()
    x = x.contiguous()
    act = torch.empty(pairs, inter, device=x.device, dtype=torch.bfloat16)
    cfg = os.environ.get("INKLING_MOEQ_CFG")
    bn, bd, nw, bh, bi, nw2 = (int(v) for v in cfg.split(",")) if cfg else (16, 128, 4, 16, 128, 4)
    _q_gate_up_kernel[(pairs, triton.cdiv(inter, bn))](
        x, w13, s13, o13, idx, act, top_k, ep_start, num_local, hidden, inter,
        x.stride(0), w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        beta, linear_beta, BN=bn, BD=bd, FMT=fmt, num_warps=nw,
    )
    y = torch.empty(pairs, hidden, device=x.device, dtype=torch.float32)
    _q_down_kernel[(pairs, triton.cdiv(hidden, bh))](
        act, w2, s2, o2, idx, wts, y, ep_start, num_local, hidden, inter,
        w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), BH=bh, BI=bi, FMT=fmt, num_warps=nw2,
    )
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)


@torch.no_grad()
def triton_moe_small_q(
    x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt,
):
    if os.environ.get("INKLING_MOEQ_PATH", "split") == "split":
        r = _moe_small_q_split(x, indices, weights, w13, s13, o13, w2, s2, o2,
                               ep_start, beta, linear_beta, fmt)
        if r is not None:
            return r
    return _moe_small_q_base(x, indices, weights, w13, s13, o13, w2, s2, o2,
                             ep_start, beta, linear_beta, fmt)


# ---- dequantise one expert's matrix to bf16 (prefill / large-batch path) ----
@triton.jit
def _dequant_kernel(
    W,
    S,
    Out,
    outer,
    K,
    stride_wn,
    stride_sn,
    BK: tl.constexpr,
    FMT: tl.constexpr,
):
    n = tl.program_id(0)
    kb = tl.program_id(1)
    if FMT == 1:
        jj = kb * (BK // 2) + tl.arange(0, BK // 2)
        jm = jj < K // 2
        pk = tl.load(W + n * stride_wn + jj, mask=jm, other=0).to(tl.int32)
        sc = _e4m3(tl.load(S + n * stride_sn + jj // 8, mask=jm, other=0).to(tl.int32))
        lo = _e2m1(pk & 15) * sc * outer
        hi = _e2m1((pk >> 4) & 15) * sc * outer
        tl.store(Out + n * K + 2 * jj, lo.to(tl.bfloat16), mask=jm)
        tl.store(Out + n * K + 2 * jj + 1, hi.to(tl.bfloat16), mask=jm)
    else:
        kk = kb * BK + tl.arange(0, BK)
        km = kk < K
        pk = tl.load(W + n * stride_wn + kk, mask=km, other=0).to(tl.int32)
        sc = tl.exp2((tl.load(S + n * stride_sn + kk // 32, mask=km, other=127).to(tl.int32) - 127).to(tl.float32))
        tl.store(Out + n * K + kk, (_e4m3(pk) * sc).to(tl.bfloat16), mask=km)


@torch.no_grad()
def dequant_expert(w: torch.Tensor, s: torch.Tensor, outer: float, fmt: int, k: int, out: torch.Tensor):
    """`w`/`s`: one expert `[R, K*]`/`[R, K/g]` -> bf16 `out [R, K]`."""
    r = w.shape[0]
    bk = 512
    _dequant_kernel[(r, triton.cdiv(k, bk))](
        w, s, out, outer, k, w.stride(0), s.stride(0), BK=bk, FMT=fmt
    )
    return out
