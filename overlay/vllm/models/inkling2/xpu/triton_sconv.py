# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton (XPU) short conv for Inkling2: paged conv-state write + taps.

Mirrors the torch `short_conv` in `sconv.py` (the numerical reference):
    conv = (sum_lag tap[lag] * x[t - lag] + x[t]) / sqrt(n_taps_scale + 1)
with the previous-token values taken from the current batch when they belong to
the same request chunk, otherwise from the paged conv-state cache.
"""

from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sconv_write_kernel(
    X,
    Cache,
    Slots,
    stride_xt,
    stride_cb,
    stride_ch,
    stride_cs,
    dim,
    width,
    block_size,
    offset,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    slot = tl.load(Slots + t)
    if slot >= 0:
        c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < dim
        x = tl.load(X + t * stride_xt + c, mask=m, other=0.0)
        head = c // width
        within = c % width
        blk = slot // block_size
        ptr = (
            Cache
            + blk.to(tl.int64) * stride_cb
            + head * stride_ch
            + (slot % block_size) * stride_cs
            + offset
            + within
        )
        tl.store(ptr, x, mask=m)


@triton.jit
def _sconv_kernel(
    X,
    Weight,
    Dyn,
    Cache,
    Pos,
    BT,
    SeqIdx,
    QStart,
    Addend,
    Residual,
    Out,
    Slots,
    inv_denom,
    stride_xt,
    stride_wc,
    stride_wk,
    stride_dt,
    stride_cb,
    stride_ch,
    stride_cs,
    stride_bt,
    stride_at,
    stride_rt,
    stride_ot,
    dim,
    width,
    block_size,
    offset,
    bt_width,
    heads,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DYNAMIC: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    m = c < dim
    head = c // width
    within = c % width
    pos = tl.load(Pos + t).to(tl.int32)
    qstart = tl.load(QStart + t)
    seq = tl.maximum(tl.load(SeqIdx + t), 0)

    x_raw = tl.load(X + t * stride_xt + c, mask=m, other=0.0)
    x_cur = x_raw.to(tl.float32)
    # fused conv-state write of this row (was a separate kernel). Rows only read the cache at positions older than
    # their request's chunk, which this launch never writes, so ordering vs. the reads below is irrelevant.
    slot = tl.load(Slots + t)
    if slot >= 0:
        tl.store(
            Cache + (slot // block_size).to(tl.int64) * stride_cb + head * stride_ch
            + (slot % block_size) * stride_cs + offset + within,
            x_raw, mask=m,
        )
    taps = tl.zeros([BLOCK_C], tl.float32)
    for lag in tl.static_range(K):
        q = pos - lag
        prev_row = t - lag
        in_chunk = (prev_row >= qstart) & (q >= 0)
        # value from the current batch
        v_b = tl.load(
            X + tl.maximum(prev_row, 0) * stride_xt + c, mask=m & in_chunk, other=0.0
        ).to(tl.float32)
        # value from the paged cache
        qc = tl.maximum(q, 0)
        lblk = tl.minimum(qc // block_size, bt_width - 1)
        blk = tl.maximum(tl.load(BT + seq * stride_bt + lblk), 0).to(tl.int64)
        v_c = tl.load(
            Cache + blk * stride_cb + head * stride_ch + (qc % block_size) * stride_cs + offset + within,
            mask=m & (~in_chunk) & (q >= 0),
            other=0.0,
        ).to(tl.float32)
        val = v_b + v_c
        if DYNAMIC:
            w = tl.load(Weight + lag * stride_wk + c, mask=m, other=0.0).to(tl.float32)
            dyn = tl.load(Dyn + t * stride_dt + head * K + lag, mask=m, other=0.0).to(tl.float32)
            w = w + dyn
        else:
            w = tl.load(Weight + c * stride_wc + (K - 1 - lag) * stride_wk, mask=m, other=0.0).to(
                tl.float32
            )
        taps += val * w
    acc = (taps + x_cur) * inv_denom
    if HAS_ADDEND:
        acc += tl.load(Addend + t * stride_at + c, mask=m, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        acc += tl.load(Residual + t * stride_rt + c, mask=m, other=0.0).to(tl.float32)
    # round to the input dtype first (the result used to be materialised in it), then widen if the output is fp32
    tl.store(Out + t * stride_ot + c, acc.to(X.dtype.element_ty).to(Out.dtype.element_ty), mask=m)


@triton.jit
def _sconv_tile_kernel(
    X, Weight, Cache, Pos, BT, SeqIdx, QStart, Slots, Addend, Residual, Out,
    inv_denom,
    stride_xt, stride_wc, stride_wk, stride_cb, stride_ch, stride_cs, stride_bt, stride_at, stride_rt, stride_ot,
    num_tokens, dim, width, block_size, offset, bt_width,
    K: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
    HAS_ADDEND: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
):
    """Static-tap conv over a BLOCK_T x BLOCK_C tile (prefill): taps loaded once per program, conv-state write fused."""
    tb = tl.program_id(0)
    cb = tl.program_id(1)
    r = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    rm = r < num_tokens
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < dim
    m2 = rm[:, None] & cm[None, :]
    head = c // width
    within = c % width
    pos = tl.load(Pos + r, mask=rm, other=0).to(tl.int32)
    qstart = tl.load(QStart + r, mask=rm, other=0)
    seq = tl.maximum(tl.load(SeqIdx + r, mask=rm, other=0), 0)
    x_cur = tl.load(X + r[:, None] * stride_xt + c[None, :], mask=m2, other=0.0)
    # fused conv-state write of this chunk's rows
    slot = tl.load(Slots + r, mask=rm, other=-1)
    wptr = (
        Cache + (slot // block_size).to(tl.int64)[:, None] * stride_cb + head[None, :] * stride_ch
        + (slot % block_size)[:, None] * stride_cs + offset + within[None, :]
    )
    tl.store(wptr, x_cur, mask=m2 & (slot >= 0)[:, None])
    taps = tl.zeros([BLOCK_T, BLOCK_C], tl.float32)
    for lag in tl.static_range(K):
        q = pos - lag
        prev_row = r - lag
        in_chunk = (prev_row >= qstart) & (q >= 0)
        v_b = tl.load(
            X + tl.maximum(prev_row, 0)[:, None] * stride_xt + c[None, :], mask=m2 & in_chunk[:, None], other=0.0
        ).to(tl.float32)
        need = rm & (~in_chunk) & (q >= 0)
        qc = tl.maximum(q, 0)
        lblk = tl.minimum(qc // block_size, bt_width - 1)
        blk = tl.maximum(tl.load(BT + seq * stride_bt + lblk, mask=need, other=0), 0).to(tl.int64)
        v_c = tl.load(
            Cache + blk[:, None] * stride_cb + head[None, :] * stride_ch + (qc % block_size)[:, None] * stride_cs + offset + within[None, :],
            mask=need[:, None] & cm[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(Weight + c * stride_wc + (K - 1 - lag) * stride_wk, mask=cm, other=0.0).to(tl.float32)
        taps += (v_b + v_c) * w[None, :]
    acc = (taps + x_cur.to(tl.float32)) * inv_denom
    if HAS_ADDEND:
        acc += tl.load(Addend + r[:, None] * stride_at + c[None, :], mask=m2, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        acc += tl.load(Residual + r[:, None] * stride_rt + c[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(Out + r[:, None] * stride_ot + c[None, :], acc.to(X.dtype.element_ty).to(Out.dtype.element_ty), mask=m2)


_TILE_MIN_T = int(os.environ.get("INKLING_SCONV_TILE_MIN_T", "128"))


@torch.no_grad()
def triton_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_idx: torch.Tensor,
    query_start: torch.Tensor,
    offset: int,
    width: int,
    *,
    dynamic_taps: torch.Tensor | None,
    heads: int,
    kernel: int,
    addend: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    num_tokens, dim = x.shape
    if x.stride(-1) != 1:  # row-strided views (e.g. splits of the fused qkvr projection) are read in place
        x = x.contiguous()
    block_size = cache.shape[2]
    out = torch.empty(num_tokens, dim, device=x.device, dtype=out_dtype or x.dtype)
    if num_tokens == 0:
        return out
    block_c = 256
    if dynamic_taps is None and num_tokens >= _TILE_MIN_T:
        add = addend if addend is not None else x
        res = residual if residual is not None else x
        block_t = 16
        _sconv_tile_kernel[(triton.cdiv(num_tokens, block_t), triton.cdiv(dim, block_c))](
            x, weight, cache, positions, block_table, seq_idx, query_start, slot_mapping, add, res, out,
            1.0 / math.sqrt(kernel + 1),
            x.stride(0), weight.stride(0), weight.stride(1), cache.stride(0), cache.stride(1), cache.stride(2),
            block_table.stride(0), add.stride(0), res.stride(0), out.stride(0),
            num_tokens, dim, width, block_size, offset, block_table.shape[1],
            K=kernel, BLOCK_C=block_c, BLOCK_T=block_t, HAS_ADDEND=addend is not None, HAS_RESIDUAL=residual is not None,
            num_warps=4,
        )
        return out
    grid = (num_tokens, triton.cdiv(dim, block_c))
    dynamic = dynamic_taps is not None
    dyn = dynamic_taps.contiguous() if dynamic else x
    pos = positions  # int64 is fine: the kernel casts on load (saves a tiny cast kernel per conv, ~190/token)
    add = addend if addend is not None else x
    res = residual if residual is not None else x
    denom = math.sqrt((2 if dynamic else 1) * kernel + 1)
    _sconv_kernel[grid](
        x, weight, dyn, cache, pos, block_table, seq_idx, query_start, add, res, out, slot_mapping,
        1.0 / denom,
        x.stride(0), weight.stride(0), weight.stride(1) if not dynamic else weight.stride(0),
        dyn.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
        block_table.stride(0), add.stride(0), res.stride(0), out.stride(0),
        dim, width, block_size, offset, block_table.shape[1], heads,
        K=kernel, BLOCK_C=block_c, DYNAMIC=dynamic,
        HAS_ADDEND=addend is not None, HAS_RESIDUAL=residual is not None,
    )
    return out
