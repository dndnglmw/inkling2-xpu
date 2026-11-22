# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton (XPU) paged attention for Inkling2: rel-bias + sink + gate + SWA.

One program per (query tile, query head). The tile's keys are streamed from the
paged KV cache (`[num_blocks, kv_heads, block_size, 2*D]`, K|V adjacent) with an
online softmax. The relative bias is gathered from the per-token table
`rel[T, H, extent]` (fp32) at `dist = q_pos - k_pos` (zero outside
`0 <= dist < extent`). The sink is a constant extra logit in the denominator and
the sigmoid gate multiplies the normalised output.

The key loop is split into a "far" region (full blocks that every row sees only
at dist >= extent: no bias gather, no mask, and a scalar block-table lookup when
BLOCK_N == block_size) and the general "near" region. The far region is most of
the iterations for long-context global layers and does not exist when
window <= extent (SWA).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_TARGET_PROGRAMS = 256  # aim for this many programs per launch
_MAX_SPLITS = 64
_UNIT_DESCALE: dict = {}


def _unit_descale(device):
    """Scalar fp32 1.0: the vendor kernel needs a scalar k/v descale tensor for an fp8 cache; ours is unit-scale."""
    d = _UNIT_DESCALE.get(device)
    if d is None:
        d = torch.ones(1, device=device, dtype=torch.float32)
        _UNIT_DESCALE[device] = d
    return d


_FAR_FOLD = os.environ.get("INKLING_XPU_FAR_FOLD", "1") == "1"
_USE_VENDOR_FAR =os.environ.get("INKLING_XPU_VENDOR_FAR", "1") == "1"
try:  # vendor (oneDNN-class) flash attention for the bias-free far keys
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as _vendor_fa
except Exception:  # pragma: no cover
    _vendor_fa = None
# Softmax runs in base-2: the host folds log2(e) into the qk scale so the
# score tile needs no extra multiply before `tl.exp2`.
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _inkling_attn_kernel(
    Q,
    KV,
    Rel,
    Sink,
    Gate,
    Out,
    PAcc,
    PM,
    PL,
    BT,
    SeqLens,
    QSL,
    Pos,
    TileReq,
    TileStart,
    FarLen,
    scale,
    stride_qt,
    stride_qh,
    stride_kvb,
    stride_kvh,
    stride_kvs,
    stride_relt,
    stride_relh,
    stride_sinkt,
    stride_gt,
    stride_ot,
    stride_btr,
    bt_width,
    block_size,
    extent,
    window,
    groups,
    num_tokens,
    num_heads,
    num_splits,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BN_EQ_BS: tl.constexpr,
    HAS_FAR: tl.constexpr,
    FAR_DESC: tl.constexpr,
    FINAL: tl.constexpr,
    DECODE_TILES: tl.constexpr,
    NEAR_ONLY: tl.constexpr,
    KV_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    if DECODE_TILES:  # one tile per request (all its <=16 tokens): graph-capture safe, no host tiles
        req = pid
        tstart = tl.load(QSL + pid)
    else:
        req = tl.load(TileReq + pid)
        tstart = tl.load(TileStart + pid)
    q_end = tl.load(QSL + req + 1)
    seqlen = tl.load(SeqLens + req)
    kvh = h // groups

    rows = tstart + tl.arange(0, BLOCK_M)
    rmask = rows < q_end
    qpos = tl.load(Pos + rows, mask=rmask, other=0).to(tl.int32)
    d = tl.arange(0, D)

    q = tl.load(
        Q + rows[:, None] * stride_qt + h * stride_qh + d[None, :],
        mask=rmask[:, None],
        other=0.0,
    )

    big = 2147483647
    hi = tl.minimum(tl.max(tl.where(rmask, qpos, -1), 0) + 1, seqlen)
    lo = 0
    if HAS_WINDOW:
        lo = tl.maximum(tl.min(tl.where(rmask, qpos, big), 0) - window + 1, 0)
    if NEAR_ONLY:
        # Keys older than FarLen[req] are handled (bias-free, fully visible) by the vendor
        # flash-attention kernel; this program covers only the near window.
        lo = tl.load(FarLen + req)

    qmin = tl.min(tl.where(rmask, qpos, big), 0)
    qmax = tl.max(tl.where(rmask, qpos, -1), 0)
    m_i = tl.full([BLOCK_M], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    nblk = (hi - lo + BLOCK_N - 1) // BLOCK_N
    per = (nblk + num_splits - 1) // num_splits
    n_start = lo + split * per * BLOCK_N
    n_end = tl.minimum(hi, n_start + per * BLOCK_N)

    # Far region: full key blocks that every row sees only at dist >= extent.
    # No bias gather, no mask, and (BLOCK_N == block_size) a scalar block-table
    # lookup per block instead of a per-lane gather. This is most of the
    # iterations for long-context global layers.
    if HAS_FAR:
        bias_free_hi = tl.maximum(qmin - extent + 1, 0)
        far_end = tl.minimum((bias_free_hi // BLOCK_N) * BLOCK_N, (n_end // BLOCK_N) * BLOCK_N)
        if HAS_WINDOW:
            # every (row, key) pair in a far block needs dist < window: qmax - n < window
            w_start = ((tl.maximum(qmax - window + 1, 0) + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
            far_start = tl.maximum(n_start, w_start)
        else:
            far_start = n_start
        for n in range(far_start, far_end, BLOCK_N):
            if FAR_DESC:
                # The tile is exactly one KV page (global layers start at lo=0,
                # so n is page-aligned): device tensor descriptors emit block
                # loads instead of per-element i64 address math. This cut the
                # 64K prefill from ~100 to ~60 ms.
                blk = tl.load(BT + req * stride_btr + n // block_size).to(tl.int64)
                b0 = KV + blk * stride_kvb + kvh * stride_kvh
                desc = tl.make_tensor_descriptor(
                    b0, shape=[block_size, 2 * D], strides=[stride_kvs, 1],
                    block_shape=[BLOCK_N, D],
                )
                k = desc.load([0, 0])
                v = desc.load([0, D])
                if KV_FP8:
                    k = k.to(tl.bfloat16)
                    v = v.to(tl.bfloat16)
            elif BN_EQ_BS:
                blk = tl.load(BT + req * stride_btr + n // block_size).to(tl.int64)
                base = blk * stride_kvb + kvh * stride_kvh
                off = (n % block_size + tl.arange(0, BLOCK_N)) * stride_kvs
                ptrs = KV + base + off[:, None] + d[None, :]
                k = tl.load(ptrs)
                v = tl.load(ptrs + D)
                if KV_FP8:
                    k = k.to(tl.bfloat16)
                    v = v.to(tl.bfloat16)
            else:
                kpos = n + tl.arange(0, BLOCK_N)
                blk = tl.load(BT + req * stride_btr + kpos // block_size).to(tl.int64)
                base = blk * stride_kvb + kvh * stride_kvh + (kpos % block_size) * stride_kvs
                k = tl.load(KV + base[:, None] + d[None, :])
                v = tl.load(KV + base[:, None] + D + d[None, :])
                if KV_FP8:
                    k = k.to(tl.bfloat16)
                    v = v.to(tl.bfloat16)
            s = tl.dot(q, tl.trans(k)) * scale
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    else:
        far_end = n_start

    for n in range(tl.maximum(n_start, far_end), n_end, BLOCK_N):
        kpos = n + tl.arange(0, BLOCK_N)
        kmask = kpos < n_end
        lblk = tl.minimum(kpos // block_size, bt_width - 1)
        blk = tl.load(BT + req * stride_btr + lblk, mask=kmask, other=0).to(tl.int64)
        blk = tl.maximum(blk, 0)
        base = blk * stride_kvb + kvh * stride_kvh + (kpos % block_size) * stride_kvs
        k = tl.load(KV + base[:, None] + d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(KV + base[:, None] + D + d[None, :], mask=kmask[:, None], other=0.0)
        if KV_FP8:
            k = k.to(tl.bfloat16)
            v = v.to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * scale
        # Relative bias is zero for dist >= extent (and for dist < 0, which is masked):
        # skip the gather for blocks that lie entirely outside [0, extent).
        if (qmax >= n) & (qmin - (n + BLOCK_N - 1) < extent):
            dist_b = qpos[:, None] - kpos[None, :]
            in_rel = (dist_b >= 0) & (dist_b < extent)
            didx = tl.minimum(tl.maximum(dist_b, 0), extent - 1)
            bias = tl.load(
                Rel + rows[:, None] * stride_relt + h * stride_relh + didx,
                mask=in_rel & rmask[:, None] & kmask[None, :],
                other=0.0,
            )
            s = s + bias * _LOG2E
        # Mask only blocks that are not fully visible to every row.
        full = (n + BLOCK_N <= n_end) & (n + BLOCK_N - 1 <= qmin)
        if HAS_WINDOW:
            full = full & (qmax - n < window)
        if not full:
            dist = qpos[:, None] - kpos[None, :]
            valid = (dist >= 0) & kmask[None, :] & rmask[:, None]
            if HAS_WINDOW:
                valid = valid & (dist < window)
            s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if FINAL:
        sink = tl.load(Sink + rows * stride_sinkt + h, mask=rmask, other=0.0).to(tl.float32)
        denom = l_i + tl.exp2(sink * _LOG2E - m_i)
        out = acc / denom[:, None]
        gate = tl.load(
            Gate + rows[:, None] * stride_gt + h * D + d[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        out = out * (1.0 / (1.0 + tl.exp(-gate)))
        tl.store(
            Out + rows[:, None] * stride_ot + h * D + d[None, :],
            out.to(Out.dtype.element_ty),
            mask=rmask[:, None],
        )
    else:
        part = (split * num_tokens + rows) * num_heads + h
        tl.store(PAcc + part[:, None] * D + d[None, :], acc, mask=rmask[:, None])
        tl.store(PM + part, m_i, mask=rmask)
        tl.store(PL + part, l_i, mask=rmask)


@triton.jit
def _inkling_combine_kernel(
    PAcc,
    PM,
    PL,
    Sink,
    Gate,
    Out,
    FarOut,
    FarLse,
    FarLenTok,
    stride_sinkt,
    stride_gt,
    stride_ot,
    stride_fot,
    stride_flh,
    stride_flt,
    num_tokens,
    num_heads,
    num_splits,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_FARP: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    sidx = tl.arange(0, BLOCK_S)
    smask = sidx < num_splits
    d = tl.arange(0, D)
    part = (sidx * num_tokens + t) * num_heads + h
    m = tl.load(PM + part, mask=smask, other=-1.0e30)
    l = tl.load(PL + part, mask=smask, other=0.0)
    acc = tl.load(PAcc + part[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
    big_m = tl.max(m, 0)
    if HAS_FARP:
        # vendor flash-attention partial over the far keys: (normalised out, natural-log LSE).
        # As an online-softmax partial: m = lse*log2(e), l = 1, acc = out. LSE = -inf (empty far
        # region) means no contribution (its output is NaN, so it must be masked, not multiplied).
        lse = tl.load(FarLse + h * stride_flh + t * stride_flt)
        far_ok = (lse > -1.0e29) & (tl.load(FarLenTok + t) > 0)
        m_far = tl.where(far_ok, lse * _LOG2E, -1.0e30)
        far_acc = tl.load(FarOut + t * stride_fot + h * D + d).to(tl.float32)
        far_acc = tl.where(far_ok, far_acc, 0.0)
        big_m = tl.maximum(big_m, m_far)
    w = tl.exp2(m - big_m)
    total_l = tl.sum(w * l, 0)
    total = tl.sum(acc * w[:, None], 0)
    if HAS_FARP:
        w_far = tl.where(far_ok, tl.exp2(m_far - big_m), 0.0)
        total_l = total_l + w_far
        total = total + far_acc * w_far
    sink = tl.load(Sink + t * stride_sinkt + h).to(tl.float32)
    out = total / (total_l + tl.exp2(sink * _LOG2E - big_m))
    gate = tl.load(Gate + t * stride_gt + h * D + d).to(tl.float32)
    out = out * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(Out + t * stride_ot + h * D + d, out.to(Out.dtype.element_ty))


def build_tiles(
    query_start_loc_cpu: list[int], block_m: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    reqs: list[int] = []
    starts: list[int] = []
    for r in range(len(query_start_loc_cpu) - 1):
        s, e = query_start_loc_cpu[r], query_start_loc_cpu[r + 1]
        for t in range(s, e, block_m):
            reqs.append(r)
            starts.append(t)
    return (
        torch.tensor(reqs, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int32, device=device),
    )


@torch.no_grad()
def triton_paged_attention(
    q: torch.Tensor,  # [T, H, D] bf16
    rel: torch.Tensor,  # [T, H, extent] fp32 (rel_states @ rel_proj)
    sink: torch.Tensor,  # [T, H]
    gate: torch.Tensor,  # [T, H*D] pre-sigmoid
    kv_cache: torch.Tensor,  # [nb, Hkv, bs, 2D]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,  # device, [R+1]
    query_start_loc_cpu: list[int],
    positions: torch.Tensor,
    extent: int,
    window: int | None,
    scaling: float,
    tile_cache: dict | None = None,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[1]
    max_q = max(
        (query_start_loc_cpu[i + 1] - query_start_loc_cpu[i] for i in range(len(query_start_loc_cpu) - 1)),
        default=0,
    )
    cfg = os.environ.get("INKLING_ATTN_CFG")
    cfg_bm, cfg_bn, cfg_w = (int(v) for v in cfg.split(",")) if cfg else (128, 128, 32)
    block_m = 16 if max_q <= 16 else cfg_bm
    decode_tiles = max_q <= 16
    if decode_tiles:
        # one program row per request; QSL (device) gives each request's first token
        num_tiles = len(query_start_loc_cpu) - 1
        tile_req = tile_start = query_start_loc
    else:
        if tile_cache is not None and block_m in tile_cache:
            tile_req, tile_start = tile_cache[block_m]
        else:
            tile_req, tile_start = build_tiles(query_start_loc_cpu, block_m, q.device)
            if tile_cache is not None:
                tile_cache[block_m] = (tile_req, tile_start)
        num_tiles = tile_req.numel()
    out = torch.zeros(num_tokens, num_heads * head_dim, device=q.device, dtype=torch.bfloat16)
    if num_tiles == 0:
        return out
    is_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    pos = positions  # int64 is fine: the kernel casts on load
    block_n = cfg_bn if max_q > 16 else 64
    # ---- far region (global layers): vendor flash attention, bias-free + fully visible ----
    far_out = far_lse = far_len = far_len_tok = None
    use_far = (
        _USE_VENDOR_FAR
        and window is None
        and _vendor_fa is not None
        and kv_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and block_table.shape[1] * kv_cache.shape[2] > extent + 2048
    )
    if use_far:
        n_q_r = query_start_loc[1:] - query_start_loc[:-1]
        p0 = seq_lens.to(torch.int32) - n_q_r.to(torch.int32)
        far_len = torch.clamp(p0 - extent + 1, min=0).to(torch.int32)  # keys at dist >= extent for every row
        tok = torch.arange(num_tokens, device=q.device, dtype=query_start_loc.dtype)
        req_tok = torch.searchsorted(query_start_loc[1:], tok, right=True).clamp(max=far_len.shape[0] - 1)
        far_len_tok = far_len[req_tok].contiguous()  # padded tokens map to the last request; harmless
        kv_t = kv_cache.transpose(1, 2)  # [nb, bs, Hkv, 2D] strided view of our cache (no copy)
        fa_kw = dict(
            softmax_scale=scaling, causal=False, return_softmax_lse=True,
            **({"k_descale": _unit_descale(q.device), "v_descale": _unit_descale(q.device)} if is_fp8 else {}),
        )
        max_k = block_table.shape[1] * kv_cache.shape[2]
        n_req = len(query_start_loc_cpu) - 1
        g = num_heads // num_kv_heads
        uniform = all(query_start_loc_cpu[i + 1] - query_start_loc_cpu[i] == max_q for i in range(n_req))
        if max_q > 1 and max_q <= 16 and _FAR_FOLD:
            # The vendor kernel only splits the KV sweep for q_len == 1; with q_len > 1 it runs ~serially
            # (22 ms/layer at 210K).  The far region is fully visible and bias-free, so verify rows of one
            # request share keys: fold them into the GQA group dim (one q_len=1 sweep, 1x KV read), or, for
            # ragged batches, give every row its own q_len=1 request over the same blocks.
            if uniform and max_q * g <= 48:
                t_ = max_q
                qf = q.view(n_req, t_, num_kv_heads, g, head_dim).permute(0, 2, 1, 3, 4).reshape(n_req, num_kv_heads * t_ * g, head_dim)
                o_f, l_f = _vendor_fa(
                    qf, kv_t[..., :head_dim], kv_t[..., head_dim:],
                    max_seqlen_q=1, cu_seqlens_q=torch.arange(n_req + 1, device=q.device, dtype=torch.int32),
                    max_seqlen_k=max_k, seqused_k=far_len, block_table=block_table, **fa_kw,
                )
                far_out = o_f.view(n_req, num_kv_heads, t_, g, head_dim).permute(0, 2, 1, 3, 4).reshape(num_tokens, num_heads, head_dim)
                far_lse = l_f.view(num_kv_heads, t_, g, n_req).permute(0, 2, 3, 1).reshape(num_heads, num_tokens)
            else:
                far_out, far_lse = _vendor_fa(
                    q, kv_t[..., :head_dim], kv_t[..., head_dim:],
                    max_seqlen_q=1, cu_seqlens_q=torch.arange(num_tokens + 1, device=q.device, dtype=torch.int32),
                    max_seqlen_k=max_k, seqused_k=far_len_tok, block_table=block_table[req_tok], **fa_kw,
                )
        else:
            far_out, far_lse = _vendor_fa(
                q, kv_t[..., :head_dim], kv_t[..., head_dim:],
                max_seqlen_q=max_q, cu_seqlens_q=query_start_loc,
                max_seqlen_k=max_k, seqused_k=far_len, block_table=block_table, **fa_kw,
            )
        max_ctx = extent + max_q + block_n  # the near kernel only spans ~the bias window
    else:
        max_ctx = block_table.shape[1] * kv_cache.shape[2]
    # Split the key range across programs when there are too few tiles*heads to
    # fill the GPU (decode / short chunks). Upper bound on the key range needs
    # no host sync: block-table width (or the window for SWA layers).
    if window is not None:
        max_ctx = min(max_ctx, window + block_m)
    max_nblk = max(1, (max_ctx + block_n - 1) // block_n)
    target = max(1, (_TARGET_PROGRAMS + num_tiles * num_heads - 1) // (num_tiles * num_heads))
    num_splits = max(1, min(target, _MAX_SPLITS, max_nblk))
    need_partials = num_splits > 1 or use_far
    if need_partials:
        p_acc = torch.empty(num_splits * num_tokens * num_heads * head_dim, device=q.device, dtype=torch.float32)
        p_m = torch.empty(num_splits * num_tokens * num_heads, device=q.device, dtype=torch.float32)
        p_l = torch.empty_like(p_m)
    else:
        p_acc = p_m = p_l = out
    grid = (num_tiles, num_heads, num_splits)
    _inkling_attn_kernel[grid](
        q, kv_cache, rel, sink, gate, out, p_acc, p_m, p_l,
        block_table, seq_lens, query_start_loc, pos, tile_req, tile_start,
        far_len if use_far else seq_lens,
        scaling * _LOG2E.value,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        rel.stride(0), rel.stride(1),
        sink.stride(0), gate.stride(0), out.stride(0), block_table.stride(0),
        block_table.shape[1], kv_cache.shape[2], extent,
        window if window is not None else 0,
        num_heads // num_kv_heads,
        num_tokens, num_heads, num_splits,
        D=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        HAS_WINDOW=window is not None, BN_EQ_BS=block_n == kv_cache.shape[2],
        HAS_FAR=(window is None or window > extent) and not use_far,
        FAR_DESC=window is None and block_n == kv_cache.shape[2] and kv_cache.stride(3) == 1 and not use_far and not is_fp8,
        FINAL=not need_partials, DECODE_TILES=decode_tiles, NEAR_ONLY=use_far, KV_FP8=is_fp8,
        num_warps=cfg_w if max_q > 16 else 4,
    )
    if need_partials:
        _inkling_combine_kernel[(num_tokens, num_heads)](
            p_acc, p_m, p_l, sink, gate, out,
            far_out if use_far else out, far_lse if use_far else p_m, far_len_tok if use_far else seq_lens,
            sink.stride(0), gate.stride(0), out.stride(0),
            far_out.stride(0) if use_far else 0,
            far_lse.stride(0) if use_far else 0, far_lse.stride(1) if use_far else 0,
            num_tokens, num_heads, num_splits,
            D=head_dim, BLOCK_S=triton.next_power_of_2(num_splits), HAS_FARP=use_far,
        )
    return out
