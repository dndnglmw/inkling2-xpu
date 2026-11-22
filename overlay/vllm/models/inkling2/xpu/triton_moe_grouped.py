# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped-GEMM MoE for Inkling2 on XPU (prefill / larger batches).

(token, top-k) pairs are sorted by local expert. Two execution strategies:

* bmm path (default, num_tokens > 128): pairs are bucketed per expert into a
  fixed capacity `cap` and run through oneAPI batched GEMMs on bf16-dequanted
  weights (~4x the Triton DPAS rate on B65). Pairs beyond the capacity
  (non-uniform routing) fall through to the Triton grouped kernels, so the
  result is correct for any routing distribution with no host syncs.
  The dequanted weights are cached across calls (checkpoint weights are
  static); disable with INKLING_MOE_NO_DEQ_CACHE=1.
* Triton path (INKLING_MOE_BMM=0 or small batches): two Triton kernels with
  in-tile NVFP4 / MXFP8 dequantisation (`tl.dot` on bf16 tiles):
    1. gate/up GEMM with the SITU epilogue  -> act [pairs_padded, I]
    2. down GEMM, routed weight applied, scattered back to pair order
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

import os as _os_dec

from .triton_moe_q import FMT_MXFP8, FMT_NVFP4, _tanh
from .triton_moe_q import _e2m1 as _e2m1_ref, _e4m3 as _e4m3_ref, _e2m1_fast, _e4m3_fast

# prefill dequant tile decoders: bit-exact integer variants are opt-in (A/B only; measured not faster so far)
if _os_dec.environ.get("INKLING_MOE_FAST_DECODE", "0") == "1":
    _e2m1, _e4m3 = _e2m1_fast, _e4m3_fast
else:
    _e2m1, _e4m3 = _e2m1_ref, _e4m3_ref


@triton.jit
def _deq_tile(
    W,
    S,
    rows,  # [BN]
    rmask,
    k0,
    stride_wn,
    stride_sn,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FMT: tl.constexpr,
):
    """bf16 `[BN, BK]` tile of dequantised weights (outer scale not applied)."""
    if FMT == 1:
        jj = tl.arange(0, BK // 2)
        pk = tl.load(W + rows[:, None] * stride_wn + (k0 // 2 + jj)[None, :], mask=rmask[:, None], other=0).to(
            tl.int32
        )
        vals = tl.interleave(_e2m1(pk & 15), _e2m1((pk >> 4) & 15))  # [BN, BK]
        sj = tl.arange(0, BK // 16)
        sc = _e4m3(
            tl.load(S + rows[:, None] * stride_sn + (k0 // 16 + sj)[None, :], mask=rmask[:, None], other=0).to(
                tl.int32
            )
        )
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 16)), (BN, BK))
        return (vals * sc).to(tl.bfloat16)
    else:
        kk = tl.arange(0, BK)
        pk = tl.load(W + rows[:, None] * stride_wn + (k0 + kk)[None, :], mask=rmask[:, None], other=0).to(
            tl.int32
        )
        sj = tl.arange(0, BK // 32)
        sc = tl.exp2(
            (
                tl.load(S + rows[:, None] * stride_sn + (k0 // 32 + sj)[None, :], mask=rmask[:, None], other=127).to(
                    tl.int32
                )
                - 127
            ).to(tl.float32)
        )
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 32, 32)), (BN, BK))
        return (_e4m3(pk) * sc).to(tl.bfloat16)


@triton.jit
def _g_gate_up_kernel(
    X,
    W13,
    S13,
    Outer,
    SortedIds,
    BlkExpert,
    Act,
    top_k,
    num_pairs,
    hidden,
    inter,
    stride_xt,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    beta,
    linear_beta,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FMT: tl.constexpr,
):
    b = tl.program_id(1)
    nb = tl.program_id(0)
    e_raw = tl.load(BlkExpert + b)
    e = tl.maximum(e_raw, 0).to(tl.int64)
    if e_raw >= 0:
        rows = b * BM + tl.arange(0, BM)
        pid = tl.load(SortedIds + rows)
        valid = pid < num_pairs
        tok = tl.where(valid, pid // top_k, 0)
        n = nb * BN + tl.arange(0, BN)
        nmask = n < inter
        acc_g = tl.zeros([BM, BN], tl.float32)
        acc_u = tl.zeros([BM, BN], tl.float32)
        kk = tl.arange(0, BK)
        wbase = W13 + e * stride_we
        sbase = S13 + e * stride_se
        for k0 in range(0, hidden, BK):
            a = tl.load(X + tok[:, None] * stride_xt + (k0 + kk)[None, :])
            wg = _deq_tile(wbase, sbase, n, nmask, k0, stride_wn, stride_sn, BN, BK, FMT)
            wu = _deq_tile(wbase, sbase, inter + n, nmask, k0, stride_wn, stride_sn, BN, BK, FMT)
            acc_g += tl.dot(a, tl.trans(wg))
            acc_u += tl.dot(a, tl.trans(wu))
        outer = tl.load(Outer + e)
        acc_g = acc_g * outer
        acc_u = acc_u * outer
        g = beta * _tanh(acc_g / beta) * (1.0 / (1.0 + tl.exp(-acc_g)))
        u = linear_beta * _tanh(acc_u / linear_beta)
        tl.store(Act + rows[:, None] * inter + n[None, :], (g * u).to(Act.dtype.element_ty), mask=nmask[None, :])


@triton.jit
def _g_down_kernel(
    Act,
    W2,
    S2,
    Outer,
    Wts,
    SortedIds,
    BlkExpert,
    Y,
    num_pairs,
    hidden,
    inter,
    stride_we,
    stride_wh,
    stride_se,
    stride_sh,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FMT: tl.constexpr,
):
    b = tl.program_id(1)
    hb = tl.program_id(0)
    e_raw = tl.load(BlkExpert + b)
    e = tl.maximum(e_raw, 0).to(tl.int64)
    if e_raw >= 0:
        rows = b * BM + tl.arange(0, BM)
        pid = tl.load(SortedIds + rows)
        valid = pid < num_pairs
        h = hb * BN + tl.arange(0, BN)
        hm = h < hidden
        acc = tl.zeros([BM, BN], tl.float32)
        kk = tl.arange(0, BK)
        for k0 in range(0, inter, BK):
            a = tl.load(Act + rows[:, None] * inter + (k0 + kk)[None, :])
            w = _deq_tile(W2 + e * stride_we, S2 + e * stride_se, h, hm, k0, stride_wh, stride_sh, BN, BK, FMT)
            acc += tl.dot(a, tl.trans(w))
        y = (acc * tl.load(Outer + e)).to(tl.bfloat16).to(tl.float32)
        wt = tl.load(Wts + tl.where(valid, pid, 0), mask=valid, other=0.0)
        y = y * wt[:, None]
        tl.store(Y + tl.where(valid, pid, 0)[:, None] * hidden + h[None, :], y, mask=valid[:, None] & hm[None, :])


@torch.no_grad()
def moe_align(indices: torch.Tensor, ep_start: int, num_local: int, bm: int):
    """-> (sorted_ids int32 [Npad], blk_expert int32 [NB]); no host syncs."""
    flat = indices.reshape(-1).long()
    p = flat.numel()
    e = flat - ep_start
    valid = (e >= 0) & (e < num_local)
    key = torch.where(valid, e, torch.full_like(e, num_local))
    order = torch.argsort(key, stable=True)
    skey = key[order]
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=flat.device).scatter_add_(
        0, key, torch.ones_like(key)
    )[:num_local]
    padded = ((counts + bm - 1) // bm) * bm
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    cnt_start = torch.cumsum(counts, 0) - counts
    skey_c = skey.clamp(max=num_local - 1)
    rank = torch.arange(p, device=flat.device) - cnt_start[skey_c]
    npad_ub = ((p + num_local * (bm - 1) + bm - 1) // bm) * bm
    dest = torch.where(skey < num_local, pad_start[skey_c] + rank, torch.full_like(rank, npad_ub))
    sorted_ids = torch.full((npad_ub + 1,), p, dtype=torch.int32, device=flat.device)
    sorted_ids.scatter_(0, dest, order.to(torch.int32))
    sorted_ids = sorted_ids[:npad_ub].contiguous()
    nb = npad_ub // bm
    block_start = torch.arange(nb, device=flat.device) * bm
    be = torch.searchsorted(pad_end, block_start, right=True)
    blk_expert = torch.where(be < num_local, be, torch.full_like(be, -1)).to(torch.int32)
    return sorted_ids, blk_expert


@triton.jit
def _dequant_batch_kernel(
    W,
    S,
    Out,
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
    pid = tl.program_id(0)
    kb = tl.program_id(1)
    nblk = tl.cdiv(R, BM)
    e = (pid // nblk).to(tl.int64)
    rows = (pid % nblk) * BM + tl.arange(0, BM)
    rmask = rows < R
    k0 = kb * BK
    kk = k0 + tl.arange(0, BK)
    tile = _deq_tile(W + e * stride_we, S + e * stride_se, rows, rmask, k0, stride_wn, stride_sn, BM, BK, FMT)
    out = Out + e * (R * K) + rows[:, None].to(tl.int64) * K + kk[None, :]
    tl.store(out, tile, mask=rmask[:, None] & (kk < K)[None, :])


_deq_cache: dict = {}
_deq_bytes = 0


def _deq_cache_budget() -> float:
    v = os.environ.get("INKLING_MOE_DEQ_CACHE_GB")
    if v:
        return float(v) * 2**30
    # Default OFF: the server sizes the KV cache first, so a dequant cache would either never
    # fill (free-VRAM guard) or steal the headroom long contexts need. Opt in with INKLING_MOE_DEQ_CACHE_GB.
    return 0.0


def _dequanted(w: torch.Tensor, s: torch.Tensor, fmt: int) -> torch.Tensor:
    """bf16 [E, R, K] dequantisation of packed `w [E, R, K*]`, cached.

    The cache is frozen-fill: entries are added while under the byte budget
    (and only if the device reports enough free memory) and never evicted.
    Core layers are revisited within a forward, so the hot set stays cached.
    """
    global _deq_bytes
    E, R = w.shape[0], w.shape[1]
    if fmt == FMT_NVFP4:
        k = w.shape[2] * 2
    else:
        k = w.shape[2]
    key = (w.data_ptr(), s.data_ptr(), tuple(w.shape), tuple(s.shape), fmt)
    if os.environ.get("INKLING_MOE_NO_DEQ_CACHE") != "1":
        hit = _deq_cache.get(key)
        if hit is not None:
            return hit
    out = torch.empty(E, R, k, device=w.device, dtype=torch.bfloat16)
    bm, bk = 32, 256
    _dequant_batch_kernel[(E * triton.cdiv(R, bm), triton.cdiv(k, bk))](
        w, s, out, R, k, w.stride(0), w.stride(1), s.stride(0), s.stride(1), BM=bm, BK=bk, FMT=fmt,
    )
    need = out.numel() * out.element_size()
    if os.environ.get("INKLING_MOE_NO_DEQ_CACHE") != "1" and _deq_bytes + need <= _deq_cache_budget():
        free = 0
        try:
            free, _ = torch.xpu.mem_get_info()
        except Exception:
            pass
        if free == 0 or free > need + 4 * 2**30:
            _deq_cache[key] = out
            _deq_bytes += need
    return out


@torch.no_grad()
def moe_align_cap(
    indices: torch.Tensor, ep_start: int, num_local: int, cap: int, bm_of: int
):
    """Bucket pairs per expert into a fixed capacity for the bmm path.

    Returns (tokb int64 [num_local*cap], inv int32 [pairs], sorted_ids, blk_expert):
    tokb[e*cap + r] = token of the r-th pair of local expert e (0 if none);
    inv[pair] = bmm slot (>=0), -2 (overflow pair, handled by Triton kernels),
    -1 (non-local pair, contributes zero).
    """
    top_k = indices.shape[1]
    flat = indices.reshape(-1).long()
    p = flat.numel()
    dev = flat.device
    e = flat - ep_start
    valid = (e >= 0) & (e < num_local)
    key = torch.where(valid, e, torch.full_like(e, num_local))
    order = torch.argsort(key, stable=True)
    skey = key[order]
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=dev).scatter_add_(
        0, key, torch.ones_like(key)
    )[:num_local]
    cnt_start = torch.cumsum(counts, 0) - counts
    skey_c = skey.clamp(max=num_local - 1)
    rank = torch.arange(p, device=dev) - cnt_start[skey_c]
    svalid = skey < num_local  # per sorted position; key == num_local marks non-local
    is_bmm = svalid & (rank < cap)
    is_of = svalid & (rank >= cap)
    dest_bmm = torch.where(is_bmm, skey_c * cap + rank, torch.full_like(rank, num_local * cap))
    tokb = torch.zeros(num_local * cap + 1, dtype=torch.long, device=dev)
    tokb.scatter_(0, dest_bmm, order // top_k)
    tokb = tokb[:num_local * cap].contiguous()
    inv = torch.full((p,), -1, dtype=torch.int32, device=dev)
    inv.scatter_(0, order, torch.where(is_bmm, dest_bmm, torch.where(is_of, -2, -1)).to(torch.int32))
    of_counts = (counts - cap).clamp(min=0)
    of_padded = ((of_counts + bm_of - 1) // bm_of) * bm_of
    of_end = torch.cumsum(of_padded, 0)
    of_start = of_end - of_padded
    npad_ub = ((p + num_local * (bm_of - 1) + bm_of - 1) // bm_of) * bm_of
    dest_of = torch.where(is_of, of_start[skey_c] + (rank - cap), torch.full_like(rank, npad_ub))
    sorted_ids = torch.full((npad_ub + 1,), p, dtype=torch.int32, device=dev)
    sorted_ids.scatter_(0, dest_of, order.to(torch.int32))
    sorted_ids = sorted_ids[:npad_ub].contiguous()
    nb = npad_ub // bm_of
    block_start = torch.arange(nb, device=dev) * bm_of
    be = torch.searchsorted(of_end, block_start, right=True)
    blk_expert = torch.where(be < num_local, be, torch.full_like(be, -1)).to(torch.int32)
    return tokb, inv, sorted_ids, blk_expert


@triton.jit
def _situ_kernel(
    GU,
    Outer,
    Act,
    cap,
    inter,
    beta,
    linear_beta,
    RB: tl.constexpr,
    BN: tl.constexpr,
):
    r0 = tl.program_id(0) * RB
    nb = tl.program_id(1)
    rows = r0 + tl.arange(0, RB)
    n = nb * BN + tl.arange(0, BN)
    nmask = n < inter
    outer = tl.load(Outer + rows // cap).to(tl.float32)[:, None]
    g = tl.load(GU + rows[:, None].to(tl.int64) * (2 * inter) + n[None, :], mask=nmask[None, :], other=0.0).to(
        tl.float32
    )
    u = tl.load(
        GU + rows[:, None].to(tl.int64) * (2 * inter) + inter + n[None, :], mask=nmask[None, :], other=0.0
    ).to(tl.float32)
    g = g * outer
    u = u * outer
    g = beta * _tanh(g / beta) * (1.0 / (1.0 + tl.exp(-g)))
    u = linear_beta * _tanh(u / linear_beta)
    tl.store(Act + rows[:, None].to(tl.int64) * inter + n[None, :], (g * u).to(tl.bfloat16), mask=nmask[None, :])


@triton.jit
def _bmm_out_kernel(
    YB,
    Inv,
    Wts,
    Outer,
    Y,
    cap,
    top_k,
    num_pairs,
    hidden,
    RB: tl.constexpr,
    BN: tl.constexpr,
):
    """y[pair] = (bmm out * outer2 * routed weight) for capacity-bucketed pairs."""
    p0 = tl.program_id(0) * RB
    hb = tl.program_id(1)
    pairs = p0 + tl.arange(0, RB)
    pmask = pairs < num_pairs
    slot = tl.load(Inv + pairs, mask=pmask, other=-1)
    use = slot >= 0
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
    v = (v * outer).to(tl.bfloat16).to(tl.float32) * wt  # match the Triton path's rounding point
    # slot == -2 rows belong to the Triton overflow path and must not be touched
    wr = pmask[:, None] & hm[None, :] & (slot != -2)[:, None]
    tl.store(Y + pairs[:, None].to(tl.int64) * hidden + h[None, :], v, mask=wr)


@torch.no_grad()
def triton_moe_grouped_q(
    x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt,
):
    num_tokens, hidden = x.shape
    top_k = indices.shape[1]
    num_local, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    pairs = num_tokens * top_k
    if pairs == 0:
        return torch.zeros(num_tokens, hidden, device=x.device, dtype=torch.bfloat16)
    y = torch.empty(pairs, hidden, device=x.device, dtype=torch.float32)
    cfg = os.environ.get("INKLING_MOE_CFG")
    if cfg:
        vals = [int(v) for v in cfg.split(",")]
        cbm, bn, bk, nw = vals[:4]
        nst = vals[4] if len(vals) > 4 else None
    elif num_tokens >= 1024:
        cbm, bn, bk, nw = 128, 64, 64, 16  # large prefill chunks: fill the XMX arrays
        nst = None
    else:
        cbm, bn, bk, nw = 64, 64, 64, 8
        nst = None
    bm = 16 if num_tokens <= 128 else cbm
    use_bmm = (
        os.environ.get("INKLING_MOE_BMM", "1") == "1"
        and num_tokens > 128
        and num_local > 0
    )
    x = x.contiguous()
    wts = weights.reshape(-1).float().contiguous()
    if not use_bmm:
        sorted_ids, blk_expert = moe_align(indices, ep_start, num_local, bm)
        nb_blocks = blk_expert.numel()
        act = torch.empty(sorted_ids.numel(), inter, device=x.device, dtype=torch.bfloat16)
        extra = {"num_warps": nw}
        if nst is not None:
            extra["num_stages"] = nst
        _g_gate_up_kernel[(triton.cdiv(inter, bn), nb_blocks)](
            x, w13, s13, o13, sorted_ids, blk_expert, act, top_k, pairs, hidden, inter,
            x.stride(0), w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
            beta, linear_beta, BM=bm, BN=bn, BK=bk, FMT=fmt, **extra,
        )
        y = torch.zeros(pairs, hidden, device=x.device, dtype=torch.float32)
        _g_down_kernel[(triton.cdiv(hidden, bn), nb_blocks)](
            act, w2, s2, o2, wts, sorted_ids, blk_expert, y, pairs, hidden, inter,
            w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), BM=bm, BN=bn, BK=bk, FMT=fmt, **extra,
        )
        return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)

    if os.environ.get("INKLING_MOE_TIERED", "1") == "1":
        # load-tiered buckets from the real per-expert counts: no overflow path, empty experts skipped
        from .triton_moe_tiered import triton_moe_tiered_q

        return triton_moe_tiered_q(x, indices, weights, w13, s13, o13, w2, s2, o2, ep_start, beta, linear_beta, fmt)

    # ---- bmm path: capacity bucket per expert + Triton overflow (INKLING_MOE_TIERED=0) ----
    mean = pairs / 128.0  # pairs spread over all 128 experts; local rank sees ~mean
    cap = max(64, min(2048, -(-int(mean + 4.5 * mean**0.5) // 64) * 64))
    tokb, inv, sorted_ids, blk_expert = moe_align_cap(indices, ep_start, num_local, cap, 64)
    w13d = _dequanted(w13, s13, fmt)
    w2d = _dequanted(w2, s2, fmt)
    xb = x[tokb].view(num_local, cap, hidden)
    gu = torch.bmm(xb, w13d.transpose(1, 2), out_dtype=torch.float32)  # [E, cap, 2I] fp32
    act = torch.empty(num_local * cap, inter, device=x.device, dtype=torch.bfloat16)
    _situ_kernel[(triton.cdiv(num_local * cap, 8), triton.cdiv(inter, 1024))](
        gu, o13, act, cap, inter, beta, linear_beta, RB=8, BN=1024, num_warps=8,
    )
    yb = torch.bmm(act.view(num_local, cap, inter), w2d.transpose(1, 2), out_dtype=torch.float32)
    _bmm_out_kernel[(triton.cdiv(pairs, 8), triton.cdiv(hidden, 1024))](
        yb, inv, wts, o2, y, cap, top_k, pairs, hidden, RB=8, BN=1024, num_warps=8,
    )
    # overflow pairs (routing heavier than `cap` per expert) via the Triton kernels
    nb_blocks = blk_expert.numel()
    if nb_blocks > 0:
        of_cfg = os.environ.get("INKLING_MOE_OF_CFG")
        obm, obn, obk, onw = (int(v) for v in of_cfg.split(",")) if of_cfg else (64, 64, 64, 8)
        act_of = torch.empty(sorted_ids.numel(), inter, device=x.device, dtype=torch.bfloat16)
        _g_gate_up_kernel[(triton.cdiv(inter, obn), nb_blocks)](
            x, w13, s13, o13, sorted_ids, blk_expert, act_of, top_k, pairs, hidden, inter,
            x.stride(0), w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
            beta, linear_beta, BM=obm, BN=obn, BK=obk, FMT=fmt, num_warps=onw,
        )
        _g_down_kernel[(triton.cdiv(hidden, obn), nb_blocks)](
            act_of, w2, s2, o2, wts, sorted_ids, blk_expert, y, pairs, hidden, inter,
            w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1), BM=obm, BN=obn, BK=obk, FMT=fmt, num_warps=onw,
        )
    return y.view(num_tokens, top_k, hidden).sum(1).to(torch.bfloat16)