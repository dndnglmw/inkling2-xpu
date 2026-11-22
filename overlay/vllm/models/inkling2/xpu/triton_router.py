# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE router for decode / small batches (XPU).

Replaces ~10 launches per visit (fp32 weight cast, fp32 GEMM, sigmoid, bias add, topk, gather, sum, div, mul, int cast):
    scores = sigmoid(x @ W_gate.T)            (fp32 accumulate over bf16 operands, no fp32 weight copy)
    sel    = scores + bias                    (selection only)
    idx    = top-8 of sel                     (iterative argmax; index order is irrelevant downstream)
    w      = scores[idx] / (sum + 1e-20) * route_scale
Two short kernels: the gate GEMV is split over expert blocks (a single program walking all 128x2304 weights took 47 us),
then one tiny program per token does the top-k. Outputs int32 indices and fp32 weights (no casts needed downstream).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _router_scores_kernel(
    X,
    W,
    Scores,
    D,
    stride_xt,
    E,
    EB: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0)
    eb = tl.program_id(1)
    e = eb * EB + tl.arange(0, EB)
    em = e < E
    acc = tl.zeros([EB], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dm = d < D
        x = tl.load(X + t * stride_xt + d, mask=dm, other=0.0).to(tl.float32)
        w = tl.load(W + e[:, None] * D + d[None, :], mask=em[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Scores + t * E + e, 1.0 / (1.0 + tl.exp(-acc)), mask=em)


@triton.jit
def _router_topk_kernel(
    Scores,
    Bias,
    OutIdx,
    OutW,
    route_scale,
    E: tl.constexpr,
    K: tl.constexpr,
):
    t = tl.program_id(0)
    e = tl.arange(0, E)
    scores = tl.load(Scores + t * E + e)
    sel = scores + tl.load(Bias + e)
    kv = tl.arange(0, K)
    idx_vec = tl.zeros([K], tl.int32)
    sc_vec = tl.zeros([K], tl.float32)
    for k in tl.static_range(K):
        idx = tl.argmax(sel, 0)
        sc = tl.sum(tl.where(e == idx, scores, 0.0), 0)
        idx_vec = tl.where(kv == k, idx, idx_vec)
        sc_vec = tl.where(kv == k, sc, sc_vec)
        sel = tl.where(e == idx, float("-inf"), sel)
    w_out = sc_vec / (tl.sum(sc_vec, 0) + 1e-20) * route_scale
    tl.store(OutIdx + t * K + kv, idx_vec)
    tl.store(OutW + t * K + kv, w_out)


@torch.no_grad()
def router_topk(
    x: torch.Tensor,  # [T, D] bf16
    gate_weight: torch.Tensor,  # [E, D] bf16
    gate_bias: torch.Tensor,  # [E] fp32
    route_scale: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, d = x.shape
    e = gate_weight.shape[0]
    scores = torch.empty(t, e, device=x.device, dtype=torch.float32)
    idx = torch.empty(t, top_k, device=x.device, dtype=torch.int32)
    wts = torch.empty(t, top_k, device=x.device, dtype=torch.float32)
    eb = 8  # 16 programs per token, each streaming 8 gate rows
    _router_scores_kernel[(t, triton.cdiv(e, eb))](x, gate_weight, scores, d, x.stride(0), e, EB=eb, BD=512, num_warps=4)
    _router_topk_kernel[(t,)](scores, gate_bias, idx, wts, route_scale, E=e, K=top_k, num_warps=4)
    return idx, wts
