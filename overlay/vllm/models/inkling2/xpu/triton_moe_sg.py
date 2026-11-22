# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-grouped small-batch MoE (2..16 tokens: speculative verify / short chunks), XPU.

Unlike the per-(token, slot) streaming kernel, each *active local expert* is read from memory
exactly once and applied to every token routed to it (GEMV-style FMAs on the dequantised tile,
no `tl.dot`), so verifying 8 draft tokens costs ~the distinct experts' bandwidth instead of
8x the per-token bandwidth. NVFP4 / MXFP8 weights, same numerics as `triton_moe_q`.
No host syncs; graph-capture safe (static grids, routing tables built with torch ops).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from .triton_moe_q import _e2m1, _e4m3, _tanh

TMAX = 16


@triton.jit
def _sg_gate_up(
    X, W13, S13, Outer, Count, TokOf, Act,
    hidden, inter, stride_xt, stride_we, stride_wn, stride_se, stride_sn,
    beta, linear_beta,
    BN: tl.constexpr, BD: tl.constexpr, TM: tl.constexpr, FMT: tl.constexpr,
):
    e = tl.program_id(0)
    nb = tl.program_id(1)
    cnt = tl.load(Count + e)
    if cnt > 0:
        e64 = e.to(tl.int64)
        n = nb * BN + tl.arange(0, BN)
        nmask = n < inter
        jr = tl.arange(0, TM)
        acc_g = tl.zeros([TM, BN], tl.float32)
        acc_u = tl.zeros([TM, BN], tl.float32)
        wbase = W13 + e64 * stride_we
        sbase = S13 + e64 * stride_se
        if FMT == 1:
            jj = tl.arange(0, BD // 2)
            for d0 in range(0, hidden, BD):
                pk_g = tl.load(wbase + n[:, None] * stride_wn + (d0 // 2 + jj)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
                pk_u = tl.load(wbase + (inter + n)[:, None] * stride_wn + (d0 // 2 + jj)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
                sc_g = _e4m3(tl.load(sbase + n[:, None] * stride_sn + (d0 // 16 + jj // 8)[None, :], mask=nmask[:, None], other=0).to(tl.int32))
                sc_u = _e4m3(tl.load(sbase + (inter + n)[:, None] * stride_sn + (d0 // 16 + jj // 8)[None, :], mask=nmask[:, None], other=0).to(tl.int32))
                lo_g = _e2m1(pk_g & 15) * sc_g
                hi_g = _e2m1((pk_g >> 4) & 15) * sc_g
                lo_u = _e2m1(pk_u & 15) * sc_u
                hi_u = _e2m1((pk_u >> 4) & 15) * sc_u
                for j in tl.static_range(TM):
                    if j < cnt:
                        tok = tl.load(TokOf + e * TM + j)
                        xe = tl.load(X + tok * stride_xt + d0 + 2 * jj).to(tl.float32)
                        xo = tl.load(X + tok * stride_xt + d0 + 2 * jj + 1).to(tl.float32)
                        pg = tl.sum(lo_g * xe[None, :] + hi_g * xo[None, :], 1)
                        pu = tl.sum(lo_u * xe[None, :] + hi_u * xo[None, :], 1)
                        acc_g = tl.where(jr[:, None] == j, acc_g + pg[None, :], acc_g)
                        acc_u = tl.where(jr[:, None] == j, acc_u + pu[None, :], acc_u)
        else:
            kk = tl.arange(0, BD)
            for d0 in range(0, hidden, BD):
                pk_g = tl.load(wbase + n[:, None] * stride_wn + (d0 + kk)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
                pk_u = tl.load(wbase + (inter + n)[:, None] * stride_wn + (d0 + kk)[None, :], mask=nmask[:, None], other=0).to(tl.int32)
                sc_g = tl.exp2((tl.load(sbase + n[:, None] * stride_sn + ((d0 + kk) // 32)[None, :], mask=nmask[:, None], other=127).to(tl.int32) - 127).to(tl.float32))
                sc_u = tl.exp2((tl.load(sbase + (inter + n)[:, None] * stride_sn + ((d0 + kk) // 32)[None, :], mask=nmask[:, None], other=127).to(tl.int32) - 127).to(tl.float32))
                w_g = _e4m3(pk_g) * sc_g
                w_u = _e4m3(pk_u) * sc_u
                for j in tl.static_range(TM):
                    if j < cnt:
                        tok = tl.load(TokOf + e * TM + j)
                        x = tl.load(X + tok * stride_xt + d0 + kk).to(tl.float32)
                        pg = tl.sum(w_g * x[None, :], 1)
                        pu = tl.sum(w_u * x[None, :], 1)
                        acc_g = tl.where(jr[:, None] == j, acc_g + pg[None, :], acc_g)
                        acc_u = tl.where(jr[:, None] == j, acc_u + pu[None, :], acc_u)
        outer = tl.load(Outer + e)
        for j in tl.static_range(TM):
            if j < cnt:
                g = tl.sum(tl.where(jr[:, None] == j, acc_g, 0.0), 0) * outer
                u = tl.sum(tl.where(jr[:, None] == j, acc_u, 0.0), 0) * outer
                a = beta * _tanh(g / beta) * (1.0 / (1.0 + tl.exp(-g))) * (linear_beta * _tanh(u / linear_beta))
                tl.store(Act + (e * TM + j) * inter + n, a.to(tl.bfloat16), mask=nmask)


@triton.jit
def _sg_down(
    Act, W2, S2, Outer, Count, PairOf, Wts, Y,
    hidden, inter, stride_we, stride_wh, stride_se, stride_sh,
    BH: tl.constexpr, BI: tl.constexpr, TM: tl.constexpr, FMT: tl.constexpr,
):
    e = tl.program_id(0)
    hb = tl.program_id(1)
    cnt = tl.load(Count + e)
    if cnt > 0:
        e64 = e.to(tl.int64)
        h = hb * BH + tl.arange(0, BH)
        hm = h < hidden
        jr = tl.arange(0, TM)
        acc = tl.zeros([TM, BH], tl.float32)
        wbase = W2 + e64 * stride_we
        sbase = S2 + e64 * stride_se
        if FMT == 1:
            jj = tl.arange(0, BI // 2)
            for i0 in range(0, inter, BI):
                pk = tl.load(wbase + h[:, None] * stride_wh + (i0 // 2 + jj)[None, :], mask=hm[:, None], other=0).to(tl.int32)
                sc = _e4m3(tl.load(sbase + h[:, None] * stride_sh + (i0 // 16 + jj // 8)[None, :], mask=hm[:, None], other=0).to(tl.int32))
                lo = _e2m1(pk & 15) * sc
                hi = _e2m1((pk >> 4) & 15) * sc
                for j in tl.static_range(TM):
                    if j < cnt:
                        ae = tl.load(Act + (e * TM + j) * inter + i0 + 2 * jj).to(tl.float32)
                        ao = tl.load(Act + (e * TM + j) * inter + i0 + 2 * jj + 1).to(tl.float32)
                        p = tl.sum(lo * ae[None, :] + hi * ao[None, :], 1)
                        acc = tl.where(jr[:, None] == j, acc + p[None, :], acc)
        else:
            kk = tl.arange(0, BI)
            for i0 in range(0, inter, BI):
                pk = tl.load(wbase + h[:, None] * stride_wh + (i0 + kk)[None, :], mask=hm[:, None], other=0).to(tl.int32)
                sc = tl.exp2((tl.load(sbase + h[:, None] * stride_sh + ((i0 + kk) // 32)[None, :], mask=hm[:, None], other=127).to(tl.int32) - 127).to(tl.float32))
                w = _e4m3(pk) * sc
                for j in tl.static_range(TM):
                    if j < cnt:
                        a = tl.load(Act + (e * TM + j) * inter + i0 + kk).to(tl.float32)
                        p = tl.sum(w * a[None, :], 1)
                        acc = tl.where(jr[:, None] == j, acc + p[None, :], acc)
        outer = tl.load(Outer + e)
        for j in tl.static_range(TM):
            if j < cnt:
                row = tl.sum(tl.where(jr[:, None] == j, acc, 0.0), 0) * outer
                pair = tl.load(PairOf + e * TM + j)
                y = row.to(tl.bfloat16).to(tl.float32) * tl.load(Wts + pair)
                tl.store(Y + pair * hidden + h, y, mask=hm)


@torch.no_grad()
def routing_tables(indices: torch.Tensor, ep_start: int, num_local: int):
    """(count[num_local] i32, tok_of[num_local*TMAX] i32, pair_of[num_local*TMAX] i32), sync-free."""
    p = indices.numel()
    top_k = indices.shape[1]
    flat = indices.reshape(-1).long() - ep_start
    valid = (flat >= 0) & (flat < num_local)
    onehot = ((flat[:, None] == torch.arange(num_local, device=flat.device)[None, :]) & valid[:, None]).to(torch.int32)
    csum = torch.cumsum(onehot, 0)
    rank = ((csum * onehot).sum(1) - 1).clamp(min=0).long()
    count = onehot.sum(0).to(torch.int32)
    dump = num_local * TMAX
    dest = torch.where(valid, flat.clamp(min=0) * TMAX + rank, torch.full_like(flat, dump))
    pairs = torch.arange(p, device=flat.device, dtype=torch.int32)
    tok_of = torch.zeros(dump + 1, dtype=torch.int32, device=flat.device).scatter_(0, dest, pairs // top_k)
    pair_of = torch.zeros(dump + 1, dtype=torch.int32, device=flat.device).scatter_(0, dest, pairs)
    return count, tok_of[:dump].contiguous(), pair_of[:dump].contiguous()


@torch.no_grad()
def triton_moe_small_grouped_q(x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt):
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    assert num_tokens <= TMAX
    count, tok_of, pair_of = routing_tables(indices, ep_start, num_local)
    x = x.contiguous()
    wts = weights.reshape(-1).float().contiguous()
    act = torch.empty(num_local * TMAX, inter, device=x.device, dtype=torch.bfloat16)
    bn, bd = 32, 128
    _sg_gate_up[(num_local, triton.cdiv(inter, bn))](
        x, w13, s13, o13, count, tok_of, act, hidden, inter,
        x.stride(0), w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        beta, linear_beta, BN=bn, BD=bd, TM=TMAX, FMT=fmt,
    )
    y = torch.zeros(num_tokens * top_k, hidden, device=x.device, dtype=torch.float32)
    bh, bi = 32, 128
    _sg_down[(num_local, triton.cdiv(hidden, bh))](
        act, w2, s2, o2, count, pair_of, wts, y, hidden, inter,
        w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), BH=bh, BI=bi, TM=TMAX, FMT=fmt,
    )
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)
