# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton (XPU) pre-attention elementwise ops + KV cache scatter."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rows_kernel(
    X,
    Scale,
    Tau,
    Out,
    heads,
    W,
    stride_xt,
    stride_xh,
    stride_ot,
    stride_oh,
    threshold,
    eps,
    BLOCK_W: tl.constexpr,
    MODE: tl.constexpr,  # 0: l2 clip, 1: unweighted rms norm
    HAS_SCALE: tl.constexpr,
    HAS_TAU: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.arange(0, BLOCK_W)
    m = c < W
    x = tl.load(X + t * stride_xt + h * stride_xh + c, mask=m, other=0.0).to(tl.float32)
    if HAS_SCALE:
        x = x * tl.load(Scale + h * W + c, mask=m, other=0.0).to(tl.float32)
    if MODE == 0:
        norm = tl.sqrt(tl.sum(x * x, 0))
        f = tl.minimum(threshold / (norm + 1e-6), 1.0)
        y = x * f
    else:
        ms = tl.sum(x * x, 0) / W
        y = x * (1.0 / tl.sqrt(ms + eps))
    y = y.to(tl.bfloat16)
    if HAS_TAU:
        y = (y.to(tl.float32) * tl.load(Tau + t)).to(tl.bfloat16)
    tl.store(Out + t * stride_ot + h * stride_oh + c, y, mask=m)


@torch.no_grad()
def rows_op(
    x: torch.Tensor,  # [T, H, W] (any strides in t/h, unit stride in W)
    *,
    mode: int,
    threshold: float = 0.0,
    eps: float = 0.0,
    scale: torch.Tensor | None = None,  # [H, W]
    tau: torch.Tensor | None = None,  # [T] fp32
) -> torch.Tensor:
    t, h, w = x.shape
    out = torch.empty(t, h, w, device=x.device, dtype=torch.bfloat16)
    if t == 0:
        return out
    _rows_kernel[(t, h)](
        x, scale if scale is not None else x, tau if tau is not None else x, out,
        h, w, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        threshold, eps,
        BLOCK_W=triton.next_power_of_2(w), MODE=mode,
        HAS_SCALE=scale is not None, HAS_TAU=tau is not None,
    )
    return out


@triton.jit
def _prep_kernel(
    QIn, KIn, VIn, RIn, QScale, RScale, Tau, QOut, ROut, Cache, Slots,
    H, HKV, D, DR,
    sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, sr_t, sr_h, soq_t, soq_h, sor_t, sor_h,
    stride_cb, stride_ch, stride_cs, block_size,
    q_thr, r_thr, k_thr, eps,
    BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr, HAS_TAU: tl.constexpr, FP8: tl.constexpr,
):
    """q / k / v / r row ops (same per-row arithmetic as `_rows_kernel`) in one launch; K and V go straight into the cache.
    (Branch-local variable names are unique on purpose: Triton merges names assigned in different if/else arms.)"""
    t = tl.program_id(0)
    hid = tl.program_id(1)
    if hid < H:  # q: l2-clip with per-head scale, optional tau
        cq = tl.arange(0, BLOCK_D)
        mq = cq < D
        xq = tl.load(QIn + t * sq_t + hid * sq_h + cq, mask=mq, other=0.0).to(tl.float32)
        xq = xq * tl.load(QScale + hid * D + cq, mask=mq, other=0.0).to(tl.float32)
        nq = tl.sqrt(tl.sum(xq * xq, 0))
        fq = tl.minimum(q_thr / (nq + 1e-6), 1.0)
        yq = (xq * fq).to(tl.bfloat16)
        if HAS_TAU:
            yq = (yq.to(tl.float32) * tl.load(Tau + t)).to(tl.bfloat16)
        tl.store(QOut + t * soq_t + hid * soq_h + cq, yq, mask=mq)
    elif hid < H + HKV:  # k: l2-clip -> cache
        hk = hid - H
        ck = tl.arange(0, BLOCK_D)
        mk = ck < D
        xk = tl.load(KIn + t * sk_t + hk * sk_h + ck, mask=mk, other=0.0).to(tl.float32)
        nk = tl.sqrt(tl.sum(xk * xk, 0))
        fk = tl.minimum(k_thr / (nk + 1e-6), 1.0)
        yk = (xk * fk).to(tl.bfloat16)
        slot_k = tl.load(Slots + t)
        if slot_k >= 0:
            base_k = Cache + (slot_k // block_size).to(tl.int64) * stride_cb + hk * stride_ch + (slot_k % block_size) * stride_cs
            if FP8:
                tl.store(base_k + ck, tl.minimum(tl.maximum(yk.to(tl.float32), -448.0), 448.0).to(Cache.dtype.element_ty), mask=mk)
            else:
                tl.store(base_k + ck, yk, mask=mk)
    elif hid < H + 2 * HKV:  # v: unweighted rms norm -> cache
        hv = hid - H - HKV
        cv = tl.arange(0, BLOCK_D)
        mv = cv < D
        xv = tl.load(VIn + t * sv_t + hv * sv_h + cv, mask=mv, other=0.0).to(tl.float32)
        msv = tl.sum(xv * xv, 0) / D
        yv = (xv * (1.0 / tl.sqrt(msv + eps))).to(tl.bfloat16)
        slot_v = tl.load(Slots + t)
        if slot_v >= 0:
            base_v = Cache + (slot_v // block_size).to(tl.int64) * stride_cb + hv * stride_ch + (slot_v % block_size) * stride_cs
            if FP8:
                tl.store(base_v + D + cv, tl.minimum(tl.maximum(yv.to(tl.float32), -448.0), 448.0).to(Cache.dtype.element_ty), mask=mv)
            else:
                tl.store(base_v + D + cv, yv, mask=mv)
    else:  # r: l2-clip with per-head scale, optional tau
        hr = hid - H - 2 * HKV
        cr = tl.arange(0, BLOCK_R)
        mr = cr < DR
        xr = tl.load(RIn + t * sr_t + hr * sr_h + cr, mask=mr, other=0.0).to(tl.float32)
        xr = xr * tl.load(RScale + hr * DR + cr, mask=mr, other=0.0).to(tl.float32)
        nr = tl.sqrt(tl.sum(xr * xr, 0))
        fr = tl.minimum(r_thr / (nr + 1e-6), 1.0)
        yr = (xr * fr).to(tl.bfloat16)
        if HAS_TAU:
            yr = (yr.to(tl.float32) * tl.load(Tau + t)).to(tl.bfloat16)
        tl.store(ROut + t * sor_t + hr * sor_h + cr, yr, mask=mr)


@torch.no_grad()
def prep_op(q, k, v, r, *, q_scale, r_scale, tau, q_thr, r_thr, k_thr, eps, cache, slots):
    """q/k/v/r are `[T, H*, W]` views (any t/h strides, unit stride in W). Returns (q_out, r_out) bf16; K/V are written to `cache`."""
    t, h, d = q.shape
    hkv = k.shape[1]
    dr = r.shape[2]
    q_out = torch.empty(t, h, d, device=q.device, dtype=torch.bfloat16)
    r_out = torch.empty(t, h, dr, device=q.device, dtype=torch.bfloat16)
    if t == 0:
        return q_out, r_out
    _prep_kernel[(t, 2 * h + 2 * hkv)](
        q, k, v, r, q_scale, r_scale, tau if tau is not None else q, q_out, r_out, cache, slots,
        h, hkv, d, dr,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), r.stride(0), r.stride(1),
        q_out.stride(0), q_out.stride(1), r_out.stride(0), r_out.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[2],
        q_thr, r_thr, k_thr, eps,
        BLOCK_D=triton.next_power_of_2(d), BLOCK_R=triton.next_power_of_2(dr),
        HAS_TAU=tau is not None, FP8=cache.dtype == torch.float8_e4m3fn,
    )
    return q_out, r_out


@triton.jit
def _kv_write_kernel(
    K,
    V,
    Cache,
    Slots,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_cb,
    stride_ch,
    stride_cs,
    block_size,
    D: tl.constexpr,
    FP8: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(Slots + t)
    if slot >= 0:
        d = tl.arange(0, D)
        blk = (slot // block_size).to(tl.int64)
        base = Cache + blk * stride_cb + h * stride_ch + (slot % block_size) * stride_cs
        kk = tl.load(K + t * stride_kt + h * stride_kh + d)
        vv = tl.load(V + t * stride_vt + h * stride_vh + d)
        if FP8:
            # unit-scale e4m3 cache: K is L2-clipped and V is RMS-normalised, so |x| << 448; clamp anyway
            kk = tl.minimum(tl.maximum(kk.to(tl.float32), -448.0), 448.0).to(Cache.dtype.element_ty)
            vv = tl.minimum(tl.maximum(vv.to(tl.float32), -448.0), 448.0).to(Cache.dtype.element_ty)
        tl.store(base + d, kk)
        tl.store(base + D + d, vv)


@torch.no_grad()
def kv_write(k: torch.Tensor, v: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor) -> None:
    t, h, d = k.shape
    if t == 0:
        return
    _kv_write_kernel[(t, h)](
        k, v, cache, slots, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[2], D=d,
        FP8=cache.dtype == torch.float8_e4m3fn,
    )
