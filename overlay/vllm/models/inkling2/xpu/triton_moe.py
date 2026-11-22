# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sync-free Triton (XPU) MoE for small token counts (decode / verify).

One program row per (token, top-k slot) pair streams that expert's weights
(`w13` packed `[gates; ups]`, `w2`) and fuses the SITU activation:

    gate = beta * tanh(g / beta) * sigmoid(g)
    up   = linear_beta * tanh(u / linear_beta)
    act  = gate * up                      (rounded to bf16, like the torch path)
    y    = bf16(act @ w2.T) * route_weight

Pairs whose expert is not local to this EP rank contribute zero.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tanh(x):
    # numerically safe tanh via sigmoid
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _gate_up_kernel(
    X,
    W13,
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
    beta,
    linear_beta,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    p = tl.program_id(0)  # pair = t * top_k + k
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
        for d0 in range(0, hidden, BD):
            d = d0 + tl.arange(0, BD)
            dm = d < hidden
            x = tl.load(X + t * stride_xt + d, mask=dm, other=0.0).to(tl.float32)
            wg = tl.load(
                W13 + e * stride_we + n[:, None] * stride_wn + d[None, :],
                mask=nmask[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            wu = tl.load(
                W13 + e * stride_we + (inter + n)[:, None] * stride_wn + d[None, :],
                mask=nmask[:, None] & dm[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_g += tl.sum(wg * x[None, :], 1)
            acc_u += tl.sum(wu * x[None, :], 1)
    g = beta * _tanh(acc_g / beta) * (1.0 / (1.0 + tl.exp(-acc_g)))
    u = linear_beta * _tanh(acc_u / linear_beta)
    act = g * u
    tl.store(Act + p * inter + n, act.to(Act.dtype.element_ty), mask=nmask)


@triton.jit
def _down_kernel(
    Act,
    W2,
    Idx,
    Wts,
    Y,
    ep_start,
    num_local,
    hidden,
    inter,
    stride_we,
    stride_wh,
    BH: tl.constexpr,
    BI: tl.constexpr,
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
        for i0 in range(0, inter, BI):
            i = i0 + tl.arange(0, BI)
            im = i < inter
            a = tl.load(Act + p * inter + i, mask=im, other=0.0).to(tl.float32)
            wd = tl.load(
                W2 + e * stride_we + h[:, None] * stride_wh + i[None, :],
                mask=hm[:, None] & im[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wd * a[None, :], 1)
    y = acc.to(tl.bfloat16).to(tl.float32) * w  # bf16 rounding of the expert output, like torch
    y = tl.where(valid, y, 0.0)
    tl.store(Y + p * hidden + h, y, mask=hm)


@torch.no_grad()
def triton_moe_small(
    x: torch.Tensor,  # [T, D] bf16
    indices: torch.Tensor,  # [T, K] int64 global expert ids
    weights: torch.Tensor,  # [T, K] fp32 (already * route_scale)
    w13: torch.Tensor,  # [E_local, 2I, D] packed [gates; ups]
    w2: torch.Tensor,  # [E_local, D, I]
    ep_start: int,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i, _ = w13.shape
    inter = two_i // 2
    pairs = num_tokens * top_k
    idx = indices.reshape(-1).to(torch.int32).contiguous()
    wts = weights.reshape(-1).float().contiguous()
    act = torch.empty(pairs, inter, device=x.device, dtype=torch.bfloat16)
    bn, bd = 32, 128
    _gate_up_kernel[(pairs, triton.cdiv(inter, bn))](
        x, w13, idx, act, top_k, ep_start, num_local, hidden, inter,
        x.stride(0), w13.stride(0), w13.stride(1), beta, linear_beta, BN=bn, BD=bd,
    )
    y = torch.empty(pairs, hidden, device=x.device, dtype=torch.float32)
    bh, bi = 32, 128
    _down_kernel[(pairs, triton.cdiv(hidden, bh))](
        act, w2, idx, wts, y, ep_start, num_local, hidden, inter,
        w2.stride(0), w2.stride(1), BH=bh, BI=bi,
    )
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)
