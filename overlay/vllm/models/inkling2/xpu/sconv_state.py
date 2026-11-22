# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short-conv state managed as a sliding-window KV cache (XPU).

Same layout as the NVIDIA implementation: one `InklingConvState` per decoder
layer (and per loop pass) emits a sliding-window cache whose per-head width
packs the four conv streams (K, V, attn-out, mlp-out). The conv runs
out-of-band with plain torch on XPU; the metadata builder replaces the CUDA
`seq_idx`/`query_start` helper kernel with `repeat_interleave`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig

# Stream order within the per-head packed width (contiguous sub-ranges).
_K, _V, _ATTN, _MLP = 0, 1, 2, 3


@dataclass
class InklingXPUSconvMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor  # [T] int64 flat slot of each token (-1 => skip)
    seq_idx: torch.Tensor  # [T] int32 token -> batch request
    query_start: torch.Tensor  # [T] int32 first x-row of each token's request


class InklingXPUSconvMetadataBuilder(AttentionMetadataBuilder[InklingXPUSconvMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, AttentionSpec)
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        self.query_start_buffer = torch.empty(max_num_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingXPUSconvMetadata:
        cm = common_attn_metadata
        num_actual = int(cm.query_start_loc_cpu[-1])
        num_padded = cm.slot_mapping.shape[0]
        qsl = cm.query_start_loc_cpu
        lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
        seq_idx = torch.repeat_interleave(
            torch.arange(cm.num_reqs, dtype=torch.int32), lens
        )
        query_start = torch.repeat_interleave(qsl[:-1].to(torch.int32), lens)
        self.seq_idx_buffer[:num_actual].copy_(seq_idx.to(self.seq_idx_buffer.device))
        self.query_start_buffer[:num_actual].copy_(query_start.to(self.query_start_buffer.device))
        if num_padded > num_actual:
            self.seq_idx_buffer[num_actual:num_padded].fill_(-1)
            self.query_start_buffer[num_actual:num_padded].fill_(-1)
        return InklingXPUSconvMetadata(
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping[:num_padded],
            seq_idx=self.seq_idx_buffer[:num_padded],
            query_start=self.query_start_buffer[:num_padded],
        )


class InklingXPUSconvBackend(AttentionBackend):
    """Dummy backend: the conv runs out-of-band against this cache."""

    @staticmethod
    def get_name() -> str:
        return "INKLING_XPU_SCONV_SWA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("InklingXPUSconvBackend has no attention impl")

    @staticmethod
    def get_builder_cls() -> type[InklingXPUSconvMetadataBuilder]:
        return InklingXPUSconvMetadataBuilder


class InklingConvState(nn.Module, AttentionLayerBase):
    """Per-decoder-layer owner of one sliding-window conv-state cache."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
        replicate_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= num_kv_heads, (
            f"sconv cache supports tp_size <= num_kv_heads ({num_kv_heads}), got {tp_size}"
        )
        self.num_kv_heads = num_kv_heads // tp_size
        hidden_per_head = hidden_size // (
            self.num_kv_heads if replicate_hidden else num_kv_heads
        )
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),  # K (TP-sharded with the kv heads)
            (head_dim, head_dim),  # V
            (2 * head_dim, hidden_per_head),  # attn output (replicated full width)
            (2 * head_dim + hidden_per_head, hidden_per_head),  # mlp output
        )
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        assert self._dtype == torch.bfloat16, (
            f"sconv cache supports bfloat16 only, got {self._dtype}"
        )
        static = vllm_config.compilation_config.static_forward_context
        if prefix in static:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static[prefix] = self

    def forward(self): ...

    @property
    def cache_block_size(self) -> int:
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingXPUSconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = AttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self._dtype,
        )
        return SlidingWindowManager.layer_config(spec, sliding_window=self.sliding_window)
