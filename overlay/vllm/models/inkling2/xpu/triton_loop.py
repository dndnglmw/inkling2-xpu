# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton (XPU) kernels for the Inkling2 block-attention residual mix and
the loop-controller blend. Numerical references: `loop.py` (attn_res_mix,
LoopStreamController) + `_rms_norm` in `model.py`.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

BD = 256
import os as _os

# (BLOCK_D, num_warps) of the attn-res mix kernel; override with INKLING_MIX_CFG="bd,nw" for sweeps
_MIX_CFG = tuple(int(v) for v in _os.environ.get("INKLING_MIX_CFG", "2048,16").split(","))


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


# --------------------------------------------------------------------------
# attn_res mix (+ optional weighted RMS norm)
# --------------------------------------------------------------------------
@triton.jit
def _mix_kernel(
    X,  # [S, T, D] fp32
    Qry,  # [Q, D] fp32 (pre-scaled)
    W,  # [WN, D] norm weight (NORM only)
    Out,  # [Q, T, D]
    S,
    Q,
    D,
    stride_xs,
    stride_xt,
    stride_oq,
    stride_ot,
    stride_wq,
    eps,
    eps_norm,
    SP: tl.constexpr,
    QP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM: tl.constexpr,
):
    t = tl.program_id(0)
    s_idx = tl.arange(0, SP)
    q_idx = tl.arange(0, QP)
    smask = s_idx < S
    qmask = q_idx < Q
    ss = tl.zeros([SP], tl.float32)
    dots = tl.zeros([QP, SP], tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        x = tl.load(
            X + s_idx[:, None] * stride_xs + t * stride_xt + d[None, :],
            mask=smask[:, None] & dm[None, :],
            other=0.0,
        )
        qv = tl.load(Qry + q_idx[:, None] * D + d[None, :], mask=qmask[:, None] & dm[None, :], other=0.0)
        ss += tl.sum(x * x, 1)
        dots += tl.sum(qv[:, None, :] * x[None, :, :], 2)
    inv = 1.0 / tl.sqrt(ss / D + eps)
    scores = tl.where(smask[None, :], dots * inv[None, :], float("-inf"))
    m = tl.max(scores, 1)
    p = tl.exp(scores - m[:, None])
    p = p / tl.sum(p, 1)[:, None]
    wgt = p * inv[None, :]  # [QP, SP]
    rinv = tl.zeros([QP], tl.float32)
    if NORM:
        ssq = tl.zeros([QP], tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            dm = d < D
            x = tl.load(
                X + s_idx[:, None] * stride_xs + t * stride_xt + d[None, :],
                mask=smask[:, None] & dm[None, :],
                other=0.0,
            )
            o = tl.sum(wgt[:, :, None] * x[None, :, :], 1)
            ssq += tl.sum(o * o, 1)
        rinv = 1.0 / tl.sqrt(ssq / D + eps_norm)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        x = tl.load(
            X + s_idx[:, None] * stride_xs + t * stride_xt + d[None, :],
            mask=smask[:, None] & dm[None, :],
            other=0.0,
        )
        o = tl.sum(wgt[:, :, None] * x[None, :, :], 1)
        if NORM:
            wv = tl.load(W + q_idx[:, None] * stride_wq + d[None, :], mask=qmask[:, None] & dm[None, :], other=0.0).to(
                tl.float32
            )
            o = o * rinv[:, None] * wv
        tl.store(
            Out + q_idx[:, None] * stride_oq + t * stride_ot + d[None, :],
            o.to(Out.dtype.element_ty),
            mask=qmask[:, None] & dm[None, :],
        )


@triton.jit
def _mix_mp_kernel(
    X0, X1, X2, X3, X4, X5,  # up to 6 fp32 [T, D] states, read in place (no stacked copy)
    Qry, W, Out,
    S, Q, D,
    stride_xt, stride_oq, stride_ot, stride_wq,
    eps, eps_norm,
    SP: tl.constexpr, QP: tl.constexpr, BLOCK_D: tl.constexpr, NORM: tl.constexpr,
):
    t = tl.program_id(0)
    s_idx = tl.arange(0, SP)
    q_idx = tl.arange(0, QP)
    ss = tl.zeros([SP], tl.float32)
    dots = tl.zeros([QP, SP], tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        for s in tl.static_range(SP):
            if s == 0:
                xp = X0
            elif s == 1:
                xp = X1
            elif s == 2:
                xp = X2
            elif s == 3:
                xp = X3
            elif s == 4:
                xp = X4
            else:
                xp = X5
            x = tl.load(xp + t * stride_xt + d, mask=dm & (s < S), other=0.0)
            sel = s_idx == s
            ss = tl.where(sel, ss + tl.sum(x * x, 0), ss)
            for q in tl.static_range(QP):
                qv = tl.load(Qry + q * D + d, mask=dm & (q < Q), other=0.0)
                dots = tl.where(sel[None, :] & (q_idx[:, None] == q), dots + tl.sum(qv * x, 0), dots)
    smask = s_idx < S
    qmask = q_idx < Q
    inv = 1.0 / tl.sqrt(ss / D + eps)
    scores = tl.where(smask[None, :], dots * inv[None, :], float("-inf"))
    m = tl.max(scores, 1)
    p = tl.exp(scores - m[:, None])
    p = p / tl.sum(p, 1)[:, None]
    wgt = p * inv[None, :]
    rinv = tl.zeros([QP], tl.float32)
    if NORM:
        ssq = tl.zeros([QP], tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            dm = d < D
            o = tl.zeros([QP, BLOCK_D], tl.float32)
            for s in tl.static_range(SP):
                if s == 0:
                    xp = X0
                elif s == 1:
                    xp = X1
                elif s == 2:
                    xp = X2
                elif s == 3:
                    xp = X3
                elif s == 4:
                    xp = X4
                else:
                    xp = X5
                x = tl.load(xp + t * stride_xt + d, mask=dm & (s < S), other=0.0)
                wcol = tl.sum(tl.where((s_idx == s)[None, :], wgt, 0.0), 1)
                o += wcol[:, None] * x[None, :]
            ssq += tl.sum(o * o, 1)
        rinv = 1.0 / tl.sqrt(ssq / D + eps_norm)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        o = tl.zeros([QP, BLOCK_D], tl.float32)
        for s in tl.static_range(SP):
            if s == 0:
                xp = X0
            elif s == 1:
                xp = X1
            elif s == 2:
                xp = X2
            elif s == 3:
                xp = X3
            elif s == 4:
                xp = X4
            else:
                xp = X5
            x = tl.load(xp + t * stride_xt + d, mask=dm & (s < S), other=0.0)
            wcol = tl.sum(tl.where((s_idx == s)[None, :], wgt, 0.0), 1)
            o += wcol[:, None] * x[None, :]
        if NORM:
            wv = tl.load(W + q_idx[:, None] * stride_wq + d[None, :], mask=qmask[:, None] & dm[None, :], other=0.0).to(tl.float32)
            o = o * rinv[:, None] * wv
        tl.store(Out + q_idx[:, None] * stride_oq + t * stride_ot + d[None, :], o.to(Out.dtype.element_ty), mask=qmask[:, None] & dm[None, :])


_MP_MIN_T = int(_os.environ.get("INKLING_MIX_MP_MIN_T", "128"))
# NOTE: a decode-size no-stack variant of the mix kernel was tried and dropped: it is not bit-identical to the stacked
# kernel (different reduction layout), and the stack itself is only one launch per call.
_MP_BD = int(_os.environ.get("INKLING_MIX_MP_BD", "1024"))
_MP_NW = int(_os.environ.get("INKLING_MIX_MP_NW", "4"))


@torch.no_grad()
def _mix_op_mp(blocks, cur, query, norm_weight, eps, norm_eps):
    states = [*blocks, cur]
    s = len(states)
    t, d = cur.shape
    q = query.shape[0]
    out = torch.empty(q, t, d, device=cur.device, dtype=torch.bfloat16 if norm_weight is not None else torch.float32)
    w = norm_weight if norm_weight is not None else query
    stride_wq = (w.stride(0) if w.shape[0] > 1 else 0) if norm_weight is not None else 0
    ptrs = states + [states[0]] * (6 - s)
    _mix_mp_kernel[(t,)](
        *ptrs, query.contiguous(), w, out, s, q, d,
        cur.stride(0), out.stride(0), out.stride(1), stride_wq, eps, norm_eps,
        SP=triton.next_power_of_2(s), QP=triton.next_power_of_2(q), BLOCK_D=_MP_BD, NORM=norm_weight is not None, num_warps=_MP_NW,
    )
    return out


@torch.no_grad()
def mix_op(
    blocks: list[torch.Tensor],
    cur: torch.Tensor,
    query: torch.Tensor,  # [Q, D] fp32 pre-scaled
    *,
    norm_weight: torch.Tensor | None = None,  # [WN, D]; WN == 1 broadcasts over queries
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
    allow_mp: bool = True,
) -> torch.Tensor:
    """fp32 `[Q, T, D]` (or bf16 normed when `norm_weight` is given)."""
    if (
        allow_mp
        and cur.shape[0] >= _MP_MIN_T
        and len(blocks) + 1 <= 6
        and all(b.dtype == torch.float32 and b.is_contiguous() and b.shape == cur.shape for b in blocks)
        and cur.dtype == torch.float32
        and cur.is_contiguous()
    ):
        return _mix_op_mp(blocks, cur, query, norm_weight, eps, norm_eps)
    states = torch.stack([*blocks, cur], 0).float().contiguous()
    s, t, d = states.shape
    q = query.shape[0]
    out = torch.empty(
        q, t, d, device=cur.device, dtype=torch.bfloat16 if norm_weight is not None else torch.float32
    )
    if t == 0:
        return out
    w = norm_weight if norm_weight is not None else query
    stride_wq = (w.stride(0) if w.shape[0] > 1 else 0) if norm_weight is not None else 0
    _bd, _nw = _MIX_CFG
    _mix_kernel[(t,)](
        states, query.contiguous(), w, out, s, q, d,
        states.stride(0), states.stride(1), out.stride(0), out.stride(1), stride_wq,
        eps, norm_eps,
        SP=triton.next_power_of_2(s), QP=triton.next_power_of_2(q), BLOCK_D=_bd,
        NORM=norm_weight is not None, num_warps=_nw,
    )
    return out


# --------------------------------------------------------------------------
# loop controller blend (+ weighted RMS norm), J streams at once
# --------------------------------------------------------------------------
@triton.jit
def _ctrl_hidden_kernel(
    Dest,  # [J, T, D] fp32 (strides arbitrary)
    Msg,  # [J, T, D] fp32
    Wdown,  # [J, R, 2D]
    Hid,  # [J, T, R] bf16
    T,
    D,
    R,
    stride_dj,
    stride_dt,
    stride_mj,
    stride_mt,
    stride_wj,
    stride_wr,
    stride_hj,
    stride_ht,
    eps,
    RB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.program_id(1)
    rb = tl.program_id(2)
    ssd = tl.zeros([BLOCK_D], tl.float32)
    ssm = tl.zeros([BLOCK_D], tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        a = tl.load(Dest + j * stride_dj + t * stride_dt + d, mask=dm, other=0.0)
        b = tl.load(Msg + j * stride_mj + t * stride_mt + d, mask=dm, other=0.0)
        ssd += a * a
        ssm += b * b
    rd = 1.0 / tl.sqrt(tl.sum(ssd, 0) / D + eps)
    rm = 1.0 / tl.sqrt(tl.sum(ssm, 0) / D + eps)
    r = rb * RB + tl.arange(0, RB)
    rmask = r < R
    acc = tl.zeros([RB], tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        a = tl.load(Dest + j * stride_dj + t * stride_dt + d, mask=dm, other=0.0)
        b = tl.load(Msg + j * stride_mj + t * stride_mt + d, mask=dm, other=0.0)
        a = (a * rd).to(tl.bfloat16).to(tl.float32)
        b = (b * rm).to(tl.bfloat16).to(tl.float32)
        w1 = tl.load(
            Wdown + j * stride_wj + r[:, None] * stride_wr + d[None, :],
            mask=rmask[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            Wdown + j * stride_wj + r[:, None] * stride_wr + D + d[None, :],
            mask=rmask[:, None] & dm[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w1 * a[None, :], 1) + tl.sum(w2 * b[None, :], 1)
    tl.store(Hid + j * stride_hj + t * stride_ht + r, acc.to(tl.bfloat16), mask=rmask)


@triton.jit
def _ctrl_blend_kernel(
    Dest,
    Msg,
    Hid,
    Wgain,  # [J, D, R]
    Wforget,  # [J, D, R]
    Out,  # [J, T, D] fp32
    D,
    R,
    stride_dj,
    stride_dt,
    stride_mj,
    stride_mt,
    stride_hj,
    stride_ht,
    stride_gj,
    stride_gd,
    stride_oj,
    stride_ot,
    BN: tl.constexpr,
    RP: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.program_id(1)
    nb = tl.program_id(2)
    n = nb * BN + tl.arange(0, BN)
    nm = n < D
    r = tl.arange(0, RP)
    rm = r < R
    h = tl.load(Hid + j * stride_hj + t * stride_ht + r, mask=rm, other=0.0).to(tl.float32)
    wg = tl.load(Wgain + j * stride_gj + n[:, None] * stride_gd + r[None, :], mask=nm[:, None] & rm[None, :], other=0.0).to(
        tl.float32
    )
    wf = tl.load(
        Wforget + j * stride_gj + n[:, None] * stride_gd + r[None, :], mask=nm[:, None] & rm[None, :], other=0.0
    ).to(tl.float32)
    g = tl.sum(wg * h[None, :], 1)
    f = tl.sum(wf * h[None, :], 1)
    dest = tl.load(Dest + j * stride_dj + t * stride_dt + n, mask=nm, other=0.0)
    msg = tl.load(Msg + j * stride_mj + t * stride_mt + n, mask=nm, other=0.0)
    out = (1.0 - _tanh(f)) * dest + _tanh(g) * msg
    tl.store(Out + j * stride_oj + t * stride_ot + n, out, mask=nm)


@triton.jit
def _wnorm_kernel(
    X,  # [J, T, D] fp32
    W,  # [WN, D]
    Out,  # [J, T, D] bf16
    D,
    stride_xj,
    stride_xt,
    stride_oj,
    stride_ot,
    stride_wj,
    eps,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.program_id(1)
    ss = tl.zeros([BLOCK_D], tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X + j * stride_xj + t * stride_xt + d, mask=d < D, other=0.0)
        ss += x * x
    rinv = 1.0 / tl.sqrt(tl.sum(ss, 0) / D + eps)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        x = tl.load(X + j * stride_xj + t * stride_xt + d, mask=dm, other=0.0)
        w = tl.load(W + j * stride_wj + d, mask=dm, other=0.0).to(tl.float32)
        tl.store(Out + j * stride_oj + t * stride_ot + d, (x * rinv * w).to(tl.bfloat16), mask=dm)


@triton.jit
def _blend_norm_kernel(
    Dest, Msg, GF, W, Out,
    D,
    stride_dj, stride_dt, stride_mj, stride_mt, stride_gj, stride_gt, stride_oj, stride_ot, stride_wj,
    norm_eps,
    BLOCK_D: tl.constexpr,
):
    """out = wnorm((1 - tanh f) * dest + tanh g * msg), g/f from the fused gain|forget GEMM; one pass per row."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    g = tl.load(GF + j * stride_gj + t * stride_gt + d, mask=dm, other=0.0)
    f = tl.load(GF + j * stride_gj + t * stride_gt + D + d, mask=dm, other=0.0)
    dest = tl.load(Dest + j * stride_dj + t * stride_dt + d, mask=dm, other=0.0)
    msg = tl.load(Msg + j * stride_mj + t * stride_mt + d, mask=dm, other=0.0)
    x = (1.0 - _tanh(f)) * dest + _tanh(g) * msg
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + norm_eps)
    w = tl.load(W + j * stride_wj + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(Out + j * stride_oj + t * stride_ot + d, (x * rinv * w).to(tl.bfloat16), mask=dm)


@triton.jit
def _ctrl_prep_kernel(
    Dest, Msg, X, D, stride_dj, stride_dt, stride_mj, stride_mt, stride_xj, stride_xt, eps,
    BLOCK_D: tl.constexpr,
):
    """x[j, t] = bf16([dest * rsqrt(mean(dest^2)+eps) | msg * rsqrt(mean(msg^2)+eps)])."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    a = tl.load(Dest + j * stride_dj + t * stride_dt + d, mask=dm, other=0.0)
    b = tl.load(Msg + j * stride_mj + t * stride_mt + d, mask=dm, other=0.0)
    ra = 1.0 / tl.sqrt(tl.sum(a * a, 0) / D + eps)
    rb = 1.0 / tl.sqrt(tl.sum(b * b, 0) / D + eps)
    tl.store(X + j * stride_xj + t * stride_xt + d, (a * ra).to(tl.bfloat16), mask=dm)
    tl.store(X + j * stride_xj + t * stride_xt + D + d, (b * rb).to(tl.bfloat16), mask=dm)


_LARGE_T = int(_os.environ.get("INKLING_CTRL_GEMM_MIN_T", "128"))  # prefill chunks use the GEMM path; decode keeps the GEMV kernels


@torch.no_grad()
def _ctrl_blend_norm_gemm(dest, msg, w_down, w_gain, w_forget, norm_weight, eps, norm_eps):
    """Large-T path: the per-token GEMVs become bmm (XMX) with fp32 accumulation/outputs, then one fused blend+norm kernel."""
    j, t, d = dest.shape
    x = torch.empty(j, t, 2 * d, device=dest.device, dtype=torch.bfloat16)
    _ctrl_prep_kernel[(t, j)](
        dest, msg, x, d, dest.stride(0), dest.stride(1), msg.stride(0), msg.stride(1), x.stride(0), x.stride(1), eps,
        BLOCK_D=triton.next_power_of_2(d), num_warps=16,
    )
    hid = torch.bmm(x, w_down.transpose(1, 2))  # [J, T, R] bf16 (fp32 accumulate)
    gf = torch.bmm(hid, torch.cat([w_gain, w_forget], 1).transpose(1, 2), out_dtype=torch.float32)  # [J, T, 2D] fp32
    out = torch.empty(j, t, d, device=dest.device, dtype=torch.bfloat16)
    _blend_norm_kernel[(t, j)](
        dest, msg, gf, norm_weight, out, d,
        dest.stride(0), dest.stride(1), msg.stride(0), msg.stride(1), gf.stride(0), gf.stride(1), out.stride(0), out.stride(1),
        norm_weight.stride(0) if norm_weight.shape[0] > 1 else 0, norm_eps,
        BLOCK_D=triton.next_power_of_2(d), num_warps=16,
    )
    return out


@torch.no_grad()
def ctrl_blend_norm_op(
    dest: torch.Tensor,  # [J, T, D] fp32 (stride 0 over J allowed)
    msg: torch.Tensor,  # [J, T, D] fp32
    w_down: torch.Tensor,  # [J, R, 2D]
    w_gain: torch.Tensor,  # [J, D, R]
    w_forget: torch.Tensor,  # [J, D, R]
    norm_weight: torch.Tensor,  # [WN, D] (WN == 1 broadcasts over J)
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `[J, T, D]`: norm_weight * rms(blend(dest, msg))."""
    j, t, d = dest.shape
    if t >= _LARGE_T:
        return _ctrl_blend_norm_gemm(dest, msg, w_down, w_gain, w_forget, norm_weight, eps, norm_eps)
    r = w_down.shape[1]
    out_bf = torch.empty(j, t, d, device=dest.device, dtype=torch.bfloat16)
    if t == 0:
        return out_bf
    hid = torch.empty(j, t, r, device=dest.device, dtype=torch.bfloat16)
    rb = 8
    _ctrl_hidden_kernel[(t, j, triton.cdiv(r, rb))](
        dest, msg, w_down, hid, t, d, r,
        dest.stride(0), dest.stride(1), msg.stride(0), msg.stride(1),
        w_down.stride(0), w_down.stride(1), hid.stride(0), hid.stride(1),
        eps, RB=rb, BLOCK_D=BD,
    )
    blend = torch.empty(j, t, d, device=dest.device, dtype=torch.float32)
    bn = 128
    _ctrl_blend_kernel[(t, j, triton.cdiv(d, bn))](
        dest, msg, hid, w_gain, w_forget, blend, d, r,
        dest.stride(0), dest.stride(1), msg.stride(0), msg.stride(1),
        hid.stride(0), hid.stride(1), w_gain.stride(0), w_gain.stride(1),
        blend.stride(0), blend.stride(1), BN=bn, RP=triton.next_power_of_2(r),
    )
    _wnorm_kernel[(t, j)](
        blend, norm_weight, out_bf, d,
        blend.stride(0), blend.stride(1), out_bf.stride(0), out_bf.stride(1),
        norm_weight.stride(0) if norm_weight.shape[0] > 1 else 0, norm_eps, BLOCK_D=BD,
    )
    return out_bf
