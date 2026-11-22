# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-tiered grouped MoE for prefill (XPU): no overflow path, hot-set dequant.

Real routing is heavy-tailed (measured on 48 real visits at T=2048: median expert gets half the mean load, the top 1%
of experts get >=1024 pairs, 5.5% of expert-visits are empty). A single per-expert capacity either wastes compute or
sheds a third of the pairs to a slow fallback. Here experts are bucketed into a power-of-two capacity ladder
(64, 128, 256, ...) from the *actual* counts, so every pair fits; empty experts are skipped entirely. Per tier:
dequantise only that tier's experts (once) -> bmm(gate/up) -> SITU -> bmm(down) -> scatter to pair order.

One host sync per call (the per-expert counts). Prefill only (eager); decode/verify never reach this path.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from .triton_moe_grouped import _situ_kernel
from .triton_moe_grouped import _deq_tile  # noqa: F401  (re-exported for the dequant kernel below)
from .triton_moe_q import FMT_NVFP4


@triton.jit
def _dequant_subset_kernel(
    W,
    S,
    Out,
    ExpIdx,
    R,
    K,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    BM: tl.constexpr,
    BK: tl.constexpr,
    FMT: tl.constexpr,
):
    """Out[j] = dequant(W[ExpIdx[j]]) for the selected experts (bf16 [E_t, R, K])."""
    pid = tl.program_id(0)
    kb = tl.program_id(1)
    nblk = tl.cdiv(R, BM)
    j = (pid // nblk).to(tl.int64)
    src = tl.load(ExpIdx + j).to(tl.int64)
    rows = (pid % nblk) * BM + tl.arange(0, BM)
    rmask = rows < R
    k0 = kb * BK
    kk = k0 + tl.arange(0, BK)
    tile = _deq_tile(W + src * stride_we, S + src * stride_se, rows, rmask, k0, stride_wn, stride_sn, BM, BK, FMT)
    out = Out + j * (R * K) + rows[:, None].to(tl.int64) * K + kk[None, :]
    tl.store(out, tile, mask=rmask[:, None] & (kk < K)[None, :])


@triton.jit
def _bmm_out_tier_kernel(
    YB,
    Inv,
    Wts,
    Outer,
    Y,
    cap,
    base,
    limit,
    num_pairs,
    hidden,
    RB: tl.constexpr,
    BN: tl.constexpr,
):
    """y[pair] = bf16(bmm_out * outer2) * routed weight, only for pairs whose global slot lies in THIS tier [base, base+limit)."""
    p0 = tl.program_id(0) * RB
    hb = tl.program_id(1)
    pairs = p0 + tl.arange(0, RB)
    pmask = pairs < num_pairs
    slot_g = tl.load(Inv + pairs, mask=pmask, other=-1)
    slot = slot_g - base
    use = (slot_g >= 0) & (slot >= 0) & (slot < limit)
    h = hb * BN + tl.arange(0, BN)
    hm = h < hidden
    e = tl.maximum(slot, 0) // cap
    outer = tl.load(Outer + e, mask=use, other=0.0).to(tl.float32)[:, None]
    wt = tl.load(Wts + pairs, mask=pmask & use, other=0.0)[:, None]
    v = tl.load(
        YB + tl.maximum(slot, 0)[:, None].to(tl.int64) * hidden + h[None, :],
        mask=(use[:, None] & hm[None, :]),
        other=0.0,
    ).to(tl.float32)
    v = (v * outer).to(tl.bfloat16).to(tl.float32) * wt  # same rounding point as the Triton path
    tl.store(Y + pairs[:, None].to(tl.int64) * hidden + h[None, :], v, mask=(pmask & use)[:, None] & hm[None, :])


_LADDER_MIN = 64


def _assign_tiers(counts: list[int], max_cap: int):
    """counts -> list of (cap, [expert ids]) using caps 64, 128, 256, ... (>= count); empty experts skipped."""
    tiers: dict[int, list[int]] = {}
    for e, n in enumerate(counts):
        if n <= 0:
            continue
        cap = _LADDER_MIN
        while cap < n:
            cap *= 2
        tiers.setdefault(cap, []).append(e)
    return sorted(tiers.items())


@torch.no_grad()
def triton_moe_tiered_q(x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt):
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    pairs = num_tokens * top_k
    dev = x.device
    if pairs == 0:
        return torch.zeros(num_tokens, hidden, device=dev, dtype=torch.bfloat16)
    x = x.contiguous()
    wts = weights.reshape(-1).float().contiguous()
    k13 = w13.shape[2] * 2 if fmt == FMT_NVFP4 else w13.shape[2]
    k2 = w2.shape[2] * 2 if fmt == FMT_NVFP4 else w2.shape[2]

    flat = indices.reshape(-1).long()
    e = flat - ep_start
    valid = (e >= 0) & (e < num_local)
    key = torch.where(valid, e, torch.full_like(e, num_local))
    order = torch.argsort(key, stable=True)
    skey = key[order]
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=dev).scatter_add_(0, key, torch.ones_like(key))[:num_local]
    counts_h = counts.tolist()  # the one host sync per call
    tiers = _assign_tiers(counts_h, num_tokens)
    y = torch.zeros(pairs, hidden, device=dev, dtype=torch.float32)
    if not tiers:
        return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)

    cnt_start = torch.cumsum(counts, 0) - counts
    skey_c = skey.clamp(max=num_local - 1)
    rank = torch.arange(pairs, device=dev) - cnt_start[skey_c]  # rank of each sorted pair within its expert
    svalid = skey < num_local
    tok_sorted = order // top_k

    # One global slot space for all tiers (tier t owns [base_t, base_t + n_e*cap)): every table is built once per call,
    # not once per tier (fewer launches + one H2D copy for the tables).
    e_base = [0] * num_local
    all_idx: list[int] = []
    tier_info = []
    base = 0
    for cap, exps in tiers:
        tier_info.append((cap, len(exps), base, len(all_idx)))
        for j, ex in enumerate(exps):
            e_base[ex] = base + j * cap
        all_idx += exps
        base += len(exps) * cap
    total = base
    tables = torch.tensor(e_base + all_idx, dtype=torch.long).to(dev)
    e_base_t = tables[:num_local]
    idx_all = tables[num_local:].to(torch.int32)
    gslot = e_base_t[skey_c] + rank
    tokb_all = torch.zeros(total + 1, dtype=torch.long, device=dev)
    tokb_all.scatter_(0, torch.where(svalid, gslot, torch.full_like(gslot, total)), tok_sorted)
    inv = torch.full((pairs,), -1, dtype=torch.int32, device=dev)
    inv.scatter_(0, order, torch.where(svalid, gslot, torch.full_like(gslot, -1)).to(torch.int32))
    idx_long = idx_all.long()
    o13_all = o13[idx_long].contiguous()
    o2_all = o2[idx_long].contiguous()

    bn_dq, bk_dq = 32, 256
    for cap, n_e, base, off in tier_info:
        idx_t = idx_all[off : off + n_e]
        tokb = tokb_all[base : base + n_e * cap]

        w13d = torch.empty(n_e, two_i, k13, device=dev, dtype=torch.bfloat16)
        _dequant_subset_kernel[(n_e * triton.cdiv(two_i, bn_dq), triton.cdiv(k13, bk_dq))](
            w13, s13, w13d, idx_t, two_i, k13, w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
            BM=bn_dq, BK=bk_dq, FMT=fmt,
        )
        w2d = torch.empty(n_e, hidden, k2, device=dev, dtype=torch.bfloat16)
        _dequant_subset_kernel[(n_e * triton.cdiv(hidden, bn_dq), triton.cdiv(k2, bk_dq))](
            w2, s2, w2d, idx_t, hidden, k2, w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1),
            BM=bn_dq, BK=bk_dq, FMT=fmt,
        )
        xb = x[tokb].view(n_e, cap, hidden)
        gu = torch.bmm(xb, w13d.transpose(1, 2), out_dtype=torch.float32)
        act = torch.empty(n_e * cap, inter, device=dev, dtype=torch.bfloat16)
        _situ_kernel[(triton.cdiv(n_e * cap, 8), triton.cdiv(inter, 1024))](
            gu, o13_all[off : off + n_e], act, cap, inter, beta, linear_beta, RB=8, BN=1024, num_warps=8,
        )
        yb = torch.bmm(act.view(n_e, cap, inter), w2d.transpose(1, 2), out_dtype=torch.float32)
        _bmm_out_tier_kernel[(triton.cdiv(pairs, 8), triton.cdiv(hidden, 1024))](
            yb, inv, wts, o2_all[off : off + n_e], y, cap, base, n_e * cap, pairs, hidden, RB=8, BN=1024, num_warps=8,
        )
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)
