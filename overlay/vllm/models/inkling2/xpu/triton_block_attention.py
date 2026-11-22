# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe Triton block attention for the Inkling2 DSpark draft (XPU).

Each request contributes `block` (8) query rows that attend NON-causally to the last
`window` keys of the request's paged cache (all keys when window == 0), which already
include the block's own keys. Plain GQA softmax attention: no relative bias, sink or gate.
Split-KV (flash-decoding) + combine kernel; no host syncs, static grid.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_FOLD = os.environ.get("INKLING_XPU_DRAFT_FOLD", "1") == "1"
_BN = int(os.environ.get("INKLING_DRAFT_BN", "128"))
_NW = int(os.environ.get("INKLING_DRAFT_NW", "16"))
_STAGES = int(os.environ.get("INKLING_DRAFT_STAGES", "3"))
_TARGET_PROGS = int(os.environ.get("INKLING_DRAFT_PROGS", "512"))
_MAX_SPLITS = int(os.environ.get("INKLING_DRAFT_MAX_SPLITS", "128"))

_LOG2E = 1.4426950408889634
ROWS = 16  # padded query rows per program (block <= 16)


@triton.jit
def _blk_kernel(
    Q, KV, BT, SeqLens, PAcc, PM, PL,
    scale, num_tokens, num_heads, num_splits, block_size, bt_width, groups, window,
    stride_qt, stride_qh, stride_kvb, stride_kvh, stride_kvs, stride_btr,
    D: tl.constexpr, BLOCK: tl.constexpr, BN: tl.constexpr, HAS_WINDOW: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    kvh = h // groups
    seqlen = tl.load(SeqLens + req)
    rows = req * BLOCK + tl.arange(0, 16)
    rmask = tl.arange(0, 16) < BLOCK
    d = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + h * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)
    lo = 0
    if HAS_WINDOW:
        lo = tl.maximum(seqlen - window, 0)
    nblk = (seqlen - lo + BN - 1) // BN
    per = (nblk + num_splits - 1) // num_splits
    n_start = lo + split * per * BN
    n_end = tl.minimum(seqlen, n_start + per * BN)
    m_i = tl.full([16], -1.0e30, tl.float32)
    l_i = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, D], tl.float32)
    for n in range(n_start, n_end, BN):
        kpos = n + tl.arange(0, BN)
        kmask = kpos < n_end
        lblk = tl.minimum(kpos // block_size, bt_width - 1)
        blk = tl.maximum(tl.load(BT + req * stride_btr + lblk, mask=kmask, other=0), 0).to(tl.int64)
        base = blk * stride_kvb + kvh * stride_kvh + (kpos % block_size) * stride_kvs
        k = tl.load(KV + base[:, None] + d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(KV + base[:, None] + D + d[None, :], mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    part = (split * num_tokens + rows) * num_heads + h
    tl.store(PAcc + part[:, None] * D + d[None, :], acc, mask=rmask[:, None])
    tl.store(PM + part, m_i, mask=rmask)
    tl.store(PL + part, l_i, mask=rmask)


@triton.jit
def _blk_kernel_fold(
    Q, KV, BT, SeqLens, PAcc, PM, PL,
    scale, num_tokens, num_heads, num_splits, block_size, bt_width, window,
    stride_qt, stride_qh, stride_kvb, stride_kvh, stride_kvs, stride_btr,
    D: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr, BN: tl.constexpr, HAS_WINDOW: tl.constexpr,
    ROWS_P: tl.constexpr,
):
    """One program per (request, KV head, split): the GQA group's heads are folded into the row dim
    (row = token * GROUPS + g), so each K/V tile is read once per KV head instead of once per query head."""
    req = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    seqlen = tl.load(SeqLens + req)
    r = tl.arange(0, ROWS_P)
    tok = r // GROUPS
    head = kvh * GROUPS + r % GROUPS
    rmask = r < BLOCK * GROUPS
    rows = req * BLOCK + tok
    d = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)
    lo = 0
    if HAS_WINDOW:
        lo = tl.maximum(seqlen - window, 0)
    nblk = (seqlen - lo + BN - 1) // BN
    per = (nblk + num_splits - 1) // num_splits
    n_start = lo + split * per * BN
    n_end = tl.minimum(seqlen, n_start + per * BN)
    m_i = tl.full([ROWS_P], -1.0e30, tl.float32)
    l_i = tl.zeros([ROWS_P], tl.float32)
    acc = tl.zeros([ROWS_P, D], tl.float32)
    for n in range(n_start, n_end, BN):
        kpos = n + tl.arange(0, BN)
        kmask = kpos < n_end
        lblk = tl.minimum(kpos // block_size, bt_width - 1)
        blk = tl.maximum(tl.load(BT + req * stride_btr + lblk, mask=kmask, other=0), 0).to(tl.int64)
        base = blk * stride_kvb + kvh * stride_kvh + (kpos % block_size) * stride_kvs
        k = tl.load(KV + base[:, None] + d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(KV + base[:, None] + D + d[None, :], mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    part = (split * num_tokens + rows) * num_heads + head
    tl.store(PAcc + part[:, None] * D + d[None, :], acc, mask=rmask[:, None])
    tl.store(PM + part, m_i, mask=rmask)
    tl.store(PL + part, l_i, mask=rmask)


@triton.jit
def _blk_combine(PAcc, PM, PL, Out, num_tokens, num_heads, num_splits, stride_ot,
                 D: tl.constexpr, BLOCK_S: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    sidx = tl.arange(0, BLOCK_S)
    smask = sidx < num_splits
    d = tl.arange(0, D)
    part = (sidx * num_tokens + t) * num_heads + h
    m = tl.load(PM + part, mask=smask, other=-1.0e30)
    l = tl.load(PL + part, mask=smask, other=0.0)
    acc = tl.load(PAcc + part[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
    big = tl.max(m, 0)
    w = tl.exp2(m - big)
    out = tl.sum(acc * w[:, None], 0) / tl.sum(w * l, 0)
    tl.store(Out + t * stride_ot + h * D + d, out.to(Out.dtype.element_ty))


@torch.no_grad()
def triton_block_attention(
    q: torch.Tensor,  # [T, H, D] bf16 (T = reqs * block)
    kv_cache: torch.Tensor,  # [nb, Hkv, bs, 2D]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,  # [reqs] device
    block: int,
    window: int,
    scale: float,
) -> torch.Tensor:
    t, h, d = q.shape
    reqs = t // block
    hkv = kv_cache.shape[1]
    out = torch.zeros(t, h * d, device=q.device, dtype=torch.bfloat16)
    if reqs == 0:
        return out
    bn = _BN
    max_ctx = block_table.shape[1] * kv_cache.shape[2]
    if window:
        max_ctx = min(max_ctx, window + block)
    max_nblk = max(1, (max_ctx + bn - 1) // bn)
    groups = h // hkv
    fold = _FOLD and block * groups <= 32
    progs = reqs * (hkv if fold else h)
    splits = max(1, min(_MAX_SPLITS, max_nblk, (_TARGET_PROGS + progs - 1) // progs))
    p_acc = torch.empty(splits * t * h * d, device=q.device, dtype=torch.float32)
    p_m = torch.empty(splits * t * h, device=q.device, dtype=torch.float32)
    p_l = torch.empty_like(p_m)
    if fold:
        _blk_kernel_fold[(reqs, hkv, splits)](
            q, kv_cache, block_table, seq_lens, p_acc, p_m, p_l,
            scale * _LOG2E, t, h, splits, kv_cache.shape[2], block_table.shape[1], window,
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            D=d, BLOCK=block, GROUPS=groups, BN=bn, HAS_WINDOW=bool(window), ROWS_P=32, num_warps=_NW, num_stages=_STAGES,
        )
    else:
        _blk_kernel[(reqs, h, splits)](
            q, kv_cache, block_table, seq_lens, p_acc, p_m, p_l,
            scale * _LOG2E, t, h, splits, kv_cache.shape[2], block_table.shape[1], groups, window,
            q.stride(0), q.stride(1), kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            D=d, BLOCK=block, BN=bn, HAS_WINDOW=bool(window), num_warps=8,
        )
    _blk_combine[(t, h)](p_acc, p_m, p_l, out, t, h, splits, out.stride(0),
                         D=d, BLOCK_S=triton.next_power_of_2(splits))
    return out
