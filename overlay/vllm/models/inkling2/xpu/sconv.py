# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolutions with paged sliding-window state (XPU, torch).

The state of each conv stream lives in the layer's conv-state cache row
(`offset:offset+width` of every head). Semantics match the validated reference:
taps at `lag` read the input stream `lag` steps back (from the same forward
when the token is in this chunk, otherwise from the cache), and the result is

    out = residual + addend + (conv(taps) + x) / sqrt((2 if dynamic else 1) * W + 1)

Local K/V convs use per-head dynamic taps `static_w[lag] + dyn[t, head, lag]`;
the attn/mlp convs are static. Everything accumulates in fp32 with one bf16
round at the end.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs

from .sconv_state import InklingConvState, InklingXPUSconvMetadata

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)


def conv_metadata(owner: InklingConvState) -> InklingXPUSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return attn_metadata.get(owner.prefix)  # type: ignore[return-value]


@torch.no_grad()
def short_conv(
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
    dynamic_taps: torch.Tensor | None = None,
    heads: int = 1,
    kernel: int = 4,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x` `[T, heads*width]` bf16; returns the updated hidden stream (bf16)."""
    num_tokens, dim = x.shape
    assert dim == heads * width
    device = x.device
    block_size = cache.shape[2]
    xf = x.float()
    # ---- write the current rows into the cache (slots are block*bs + pos) ----
    valid = slot_mapping >= 0
    slots = slot_mapping.long()
    rows = xf.view(num_tokens, heads, width)
    cache[slots[valid] // block_size, :, slots[valid] % block_size, offset : offset + width] = rows[
        valid
    ].to(cache.dtype)

    # ---- gather taps ----
    dyn = None
    if dynamic_taps is not None:
        dyn = dynamic_taps.float().view(num_tokens, heads, kernel)
    taps = torch.zeros(num_tokens, dim, device=device, dtype=torch.float32)
    rows_idx = torch.arange(num_tokens, device=device)
    for lag in range(kernel):
        q = positions - lag
        # query_start is the batch-row index where this token's request chunk begins;
        # the lag-th previous token lives at row (own row - lag) if it is in the chunk.
        prev_row = rows_idx - lag
        in_chunk = (prev_row >= query_start) & (q >= 0)
        row_idx = prev_row.clamp(min=0)
        val = xf[row_idx]
        qc = q.clamp(min=0)
        lblk = (qc // block_size).clamp(max=block_table.shape[1] - 1)
        block = block_table[seq_idx, lblk].long().clamp(min=0)
        cached = cache[block, :, qc % block_size, offset : offset + width].float().view(
            num_tokens, dim
        )
        val = torch.where(in_chunk[:, None], val, cached)
        val = torch.where((q >= 0)[:, None], val, torch.zeros_like(val))
        if dyn is not None:
            static = weight.float()[lag].view(1, heads, width)
            tap = (static + dyn[:, :, lag][:, :, None]).reshape(num_tokens, dim)
            taps += val * tap
        else:
            taps += val * weight.float()[:, kernel - 1 - lag]

    denom = math.sqrt((2 if dyn is not None else 1) * kernel + 1)
    conv = (taps + xf) / denom
    acc = conv
    if addend is not None:
        acc = acc + addend.float()
    if residual is not None:
        acc = acc + residual.float()
    return acc.to(x.dtype)


def _short_conv_impl(x, *args, **kwargs):
    # INKLING_XPU_TORCH_CONV=1 forces the torch reference.
    if x.device.type == "xpu" and os.environ.get("INKLING_XPU_TORCH_CONV", "0") != "1":
        from .triton_sconv import triton_short_conv

        return triton_short_conv(x, *args, **kwargs)
    kwargs.pop("out_dtype", None)
    return short_conv(x, *args, **kwargs)


class Inkling2XPUShortConv(nn.Module):
    """Static conv (attn/mlp streams) or per-head dynamic conv (K/V streams)."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.kernel_size = kernel_size
        self.dynamic = dynamic_input_size is not None
        self.num_heads = num_heads
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.dynamic:
            self.static_w = nn.Parameter(torch.empty(kernel_size, dim), requires_grad=False)
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size), requires_grad=False
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size), requires_grad=False)
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.dynamic:
            return None
        return nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
        dynamic_taps: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """`out_dtype=torch.float32` returns the bf16-rounded result widened to fp32 (what `.float()` on the bf16 output gave)."""
        assert not self.dynamic or dynamic_taps is not None
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        if m is None:
            if addend is not None:
                acc = addend.float() if residual is None else residual.float() + addend.float()
                x = (acc + x.float()).to(addend.dtype)
            if out_dtype is not None:
                x = x.to(out_dtype)
            return x if out is None else out.copy_(x)
        offset, width = owner.stream_ranges[self.stream]
        y = _short_conv_impl(
            x,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.slot_mapping,
            m.seq_idx,
            m.query_start,
            offset,
            width,
            dynamic_taps=dynamic_taps,
            heads=self.num_heads if self.dynamic else x.shape[-1] // width,
            kernel=self.kernel_size,
            addend=addend,
            residual=residual,
            **({"out_dtype": out_dtype} if out_dtype is not None and x.device.type == "xpu" else {}),
        )
        if out_dtype is not None and y.dtype != out_dtype:
            y = y.to(out_dtype)
        return y if out is None else out.copy_(y)
