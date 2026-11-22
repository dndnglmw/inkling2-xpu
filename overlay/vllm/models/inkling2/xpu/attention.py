# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention on Intel XPU.

Custom attention because Inkling2 adds three things no stock backend supports:
a per-token relative-position bias table, a per-head sink logit, and a
per-token attention-output gate. The paged KV cache and metadata come from the
V1 engine; the attention math is implemented here (torch reference path now,
Triton-XPU kernel next).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import os
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from . import kv_sim as _kv_sim
from .prof import ph as _ph
from .triton_prep import kv_write, prep_op, rows_op
from .sconv import Inkling2XPUShortConv
from .triton_attention import triton_paged_attention
from .sconv_state import InklingConvState

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)

# INKLING_XPU_TORCH_ATTN=1 forces the (slow) pure-torch attention path.
_USE_TRITON = os.environ.get("INKLING_XPU_TORCH_ATTN", "0") != "1"
_FUSED_PREP = os.environ.get("INKLING_XPU_FUSED_PREP", "1") == "1"  # q/k/v/r row ops + KV write in one launch (bit-identical)


@dataclass
class InklingXPUAttentionMetadata(AttentionMetadata):
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    num_actual_tokens: int


class InklingXPUAttentionMetadataBuilder(
    AttentionMetadataBuilder[InklingXPUAttentionMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingXPUAttentionMetadata:
        cm = common_attn_metadata
        return InklingXPUAttentionMetadata(
            block_table=cm.block_table_tensor,
            seq_lens=cm.seq_lens,
            query_start_loc=cm.query_start_loc,
            query_start_loc_cpu=cm.query_start_loc_cpu,
            slot_mapping=cm.slot_mapping,
            max_query_len=cm.max_query_len,
            num_actual_tokens=cm.num_actual_tokens,
        )


class InklingXPUAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING_XPU"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling2 XPU runs its own attention")

    @staticmethod
    def get_builder_cls() -> type[InklingXPUAttentionMetadataBuilder]:
        return InklingXPUAttentionMetadataBuilder

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True


def _kv_cache_layer_config(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
    dtype: torch.dtype | None = None,
) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        # `dtype` pins a layer's cache dtype (the DSpark draft stays bf16 even with --kv-cache-dtype fp8)
        dtype=dtype
        if dtype is not None
        else kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config),
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)


class InklingXPUAttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass (each pass attends its own K/V)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        static = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in static:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingXPUAttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


def _l2_clip(
    x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None
) -> torch.Tensor:
    x = x.float() if scale is None else x.float() * scale.float()
    norm = x.norm(dim=-1, keepdim=True)
    return (x * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(torch.bfloat16)


def _v_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)


@torch.no_grad()
def _paged_attention(
    q: torch.Tensor,  # [T, H, D] bf16
    k: torch.Tensor,  # [T, Hkv, D] bf16
    v: torch.Tensor,  # [T, Hkv, D] bf16 (value-normed)
    rel_states: torch.Tensor,  # [T, H, d_rel] bf16
    gate: torch.Tensor,  # [T, H*D] bf16 pre-sigmoid
    sink: torch.Tensor,  # [T, H] bf16 pre-sigmoid logit
    rel_proj: torch.Tensor,  # [d_rel, extent] fp32
    kv_cache: torch.Tensor,  # [nb, bs, Hkv, 2*D]
    md: InklingXPUAttentionMetadata,
    positions: torch.Tensor,
    extent: int,
    window: int | None,
    scaling: float,
    skip_kv_write: bool = False,  # K/V were already written into the cache by the fused prep kernel (k, v are then None)
) -> torch.Tensor:
    """Paged causal attention with rel bias, sink and gate -> [T, H*D] bf16."""
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[1] if k is None else k.shape[1]
    groups = num_heads // num_kv_heads
    block_size = kv_cache.shape[2]
    cache_dtype = kv_cache.dtype

    if _USE_TRITON and q.device.type == "xpu":
        if md.slot_mapping.numel() > 0 and not skip_kv_write:
            kv_write(k, v, kv_cache, md.slot_mapping)
    elif md.slot_mapping.numel() > 0:
        slots = md.slot_mapping.long()
        valid = slots >= 0
        kv_cache[slots[valid] // block_size, :, slots[valid] % block_size, :] = torch.cat(
            [k[valid].float(), v[valid].float()], dim=-1
        ).to(cache_dtype)

    rel_logits = torch.einsum("t h d, d e -> t h e", rel_states.float(), rel_proj.float())
    qsl = md.query_start_loc_cpu.tolist()
    if _USE_TRITON and q.device.type == "xpu":
        return triton_paged_attention(
            q, rel_logits.contiguous(), sink, gate, kv_cache, md.block_table, md.seq_lens,
            md.query_start_loc, qsl, positions, extent, window, scaling,
            tile_cache=md.__dict__.setdefault("_tiles", {}),
        )
    out = torch.empty(num_tokens, num_heads, head_dim, device=q.device, dtype=torch.float32)
    seq_lens = md.seq_lens.tolist()
    for r in range(len(seq_lens)):
        start, end = qsl[r], qsl[r + 1]
        if end <= start:
            continue
        seqlen = int(seq_lens[r])
        n_q = end - start
        kv_start = 0 if window is None else max(0, seqlen - window)
        kv_pos = torch.arange(kv_start, seqlen, device=q.device)
        lblk = (kv_pos // block_size).clamp(max=md.block_table.shape[1] - 1)
        blocks = md.block_table[r, lblk].long().clamp(min=0)
        kv = kv_cache[blocks, :, kv_pos % block_size, :]  # [S, Hkv, 2D]
        k_all = kv[..., :head_dim].float()
        v_all = kv[..., head_dim:].float()

        q_pos = positions[start:end].long()
        dist = q_pos[:, None] - kv_pos[None, :]
        mask = dist < 0
        if window is not None:
            mask = mask | (dist >= window)
        idx = dist.clamp(0, extent - 1)
        bias = rel_logits[start:end].float().gather(
            2, idx[:, None, :].expand(n_q, num_heads, -1)
        )
        bias = bias.masked_fill(((dist < 0) | (dist >= extent))[:, None, :], 0.0)

        kf = k_all.repeat_interleave(groups, dim=1)
        vf = v_all.repeat_interleave(groups, dim=1)
        scores = torch.einsum("q h d, s h d -> h q s", q[start:end].float(), kf) * scaling
        scores = scores + bias.permute(1, 0, 2)
        scores = scores.masked_fill(mask[None], float("-inf"))
        m = scores.amax(dim=-1, keepdim=True).clamp(min=-1e30)
        p = torch.exp(scores - m)
        sink_r = sink[start:end].float().transpose(0, 1)  # [H, n_q]
        denom = p.sum(-1) + torch.exp(sink_r - m.squeeze(-1))
        o = torch.einsum("h q s, s h d -> q h d", p, vf) / denom.transpose(0, 1)[:, :, None]
        out[start:end] = o

    out = out * torch.sigmoid(gate.float().view(num_tokens, num_heads, head_dim))
    return out.to(torch.bfloat16).reshape(num_tokens, num_heads * head_dim)


@torch.no_grad()
def _dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_states: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    rel_proj: torch.Tensor,
    positions: torch.Tensor,
    extent: int,
    window: int | None,
    scaling: float,
) -> torch.Tensor:
    """No-cache attention for profiling runs (one dense causal pass)."""
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    groups = num_heads // num_kv_heads
    q_pos = positions.long()
    dist = q_pos[:, None] - q_pos[None, :]
    mask = dist < 0
    if window is not None:
        mask = mask | (dist >= window)
    idx = dist.clamp(0, extent - 1)
    bias = torch.einsum("t h d, d e -> t h e", rel_states.float(), rel_proj.float()).gather(
        2, idx[:, None, :].expand(num_tokens, num_heads, -1)
    )
    bias = bias.masked_fill(((dist < 0) | (dist >= extent))[:, None, :], 0.0).permute(1, 0, 2)
    kf = k.float().repeat_interleave(groups, dim=1)
    vf = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("q h d, s h d -> h q s", q.float(), kf) * scaling + bias
    scores = scores.masked_fill(mask[None], float("-inf"))
    m = scores.amax(dim=-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(scores - m)
    sink_r = sink.float().transpose(0, 1)
    denom = p.sum(-1) + torch.exp(sink_r - m.squeeze(-1))
    out = torch.einsum("h q s, s h d -> q h d", p, vf) / denom.transpose(0, 1)[:, :, None]
    out = out * torch.sigmoid(gate.float().view(num_tokens, num_heads, head_dim))
    return out.to(torch.bfloat16).reshape(num_tokens, num_heads * head_dim)


_FP8_DENSE = os.environ.get("INKLING_XPU_FP8_DENSE", "0") == "1"


class InklingXPUAttention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer (XPU)."""

    def __init__(
        self,
        config,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        num_kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size if is_local else None
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            return_bias=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(torch.empty(self.num_heads, self.head_dim), requires_grad=False)
        self.r_scale = nn.Parameter(torch.empty(self.num_heads, self.d_rel), requires_grad=False)
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            InklingXPUAttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2XPUShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )
        self._fp8 = None
        self._rel_f32 = None
        self.kv_cache = torch.tensor([])
        static = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in static:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingXPUAttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    @staticmethod
    def _fp8_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-output-channel RTN e4m3: returns ([K,N] weight view, [1,N] fp32 scale)."""
        wf = w.float()
        scale = (wf.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
        q = (wf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q.contiguous().t(), scale.t().contiguous()

    def process_weights_after_loading(self) -> None:
        # constant fp32 copy of the rel-bias projection: `_paged_attention` used to re-cast it on every call (one kernel/visit)
        self._rel_f32 = self.rel_logits_proj.data.float().contiguous()
        self._fp8 = None
        if not _FP8_DENSE or self.qkvr.weight.device.type != "xpu":
            return
        w = self.qkvr.weight.data
        q_side = sum(self.split_sizes[:4])
        kv = self.split_sizes[4]
        parts = {
            "q": w[:q_side],
            "k": w[q_side : q_side + kv],
            "v": w[q_side + kv : q_side + 2 * kv],
        }
        if self.is_local:  # same-input fast path uses the fused weight
            parts["all"] = w
        packed = {k: self._fp8_pack(v) for k, v in parts.items()}
        packed["wo"] = self._fp8_pack(self.wo_ud.weight.data)
        self._fp8 = packed
        # free the bf16 copies (only the fp8 path is used from now on)
        self.qkvr.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        self.wo_ud.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)

    def _fp8_linear(self, x: torch.Tensor, key: str) -> torch.Tensor:
        w, s = self._fp8[key]
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, w, s, None)

    def wo_apply(self, x: torch.Tensor) -> torch.Tensor:
        if self._fp8 is not None:
            return self._fp8_linear(x, "wo")
        return self.wo_ud(x)

    def _project_same(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._fp8 is not None:
            if "all" not in self._fp8:
                return self._project(x, x.clone(), x.clone())
            projected = self._fp8_linear(x, "all")
        else:
            projected = nn.functional.linear(x, self.qkvr.weight)
        return list(projected.split(self.split_sizes, dim=-1)[:6])

    def _project(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> list[torch.Tensor]:
        if x_k is x_q and x_v is x_q:
            return self._project_same(x_q)
        if self._fp8 is not None:
            q4 = self._fp8_linear(x_q, "q").split(self.split_sizes[:4], dim=-1)
            return [*q4, self._fp8_linear(x_k, "k"), self._fp8_linear(x_v, "v")]
        weight = self.qkvr.weight
        q_side = sum(self.split_sizes[:4])  # K rows start here (pad is last in the fused weight)
        kv_size = self.split_sizes[4]
        k_end = q_side + kv_size
        q4 = nn.functional.linear(x_q, weight[:q_side]).split(self.split_sizes[:4], dim=-1)
        k = nn.functional.linear(x_k, weight[q_side:k_end])
        v = nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*q4, k, v]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        num_tokens = x_q.shape[0]

        with _ph("a.proj"):
            q, r, gate, sink, k_proj, v_proj = self._project(x_q, x_k, x_v)
        owner = self.conv_states[pass_idx] if pass_idx < len(self.conv_states) else self.conv_state
        with _ph("a.kvconv"):
            dyn_k = self.k_sconv.dynamic_taps(x_k) if not self.is_local else None
            dyn_v = self.v_sconv.dynamic_taps(x_v) if not self.is_local else None
            k = self.k_sconv(k_proj, positions, owner=owner, dynamic_taps=dyn_k)
            v = self.v_sconv(v_proj, positions, owner=owner, dynamic_taps=dyn_v)

        attn_md = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        if kv_cache.dtype == torch.uint8:  # --kv-cache-dtype fp8: raw bytes; unit-scale e4m3 (K clipped, V RMS-normalised)
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        has_cache = isinstance(attn_md, dict) and kv_cache.numel() != 0
        fused = (
            _FUSED_PREP and _USE_TRITON and q.device.type == "xpu" and has_cache and _kv_sim.MODE[0] == "off"
            and attn_md[layer_name].slot_mapping.numel() > 0
        )

        with _ph("a.norms"):
            if fused:
                # one launch: q/r normalised, K/V normalised and written straight into the paged cache (bit-identical)
                q, r = prep_op(
                    q.view(num_tokens, self.num_heads, self.head_dim), k.view(num_tokens, self.num_kv_heads, self.head_dim),
                    v.view(num_tokens, self.num_kv_heads, self.head_dim), r.view(num_tokens, self.num_heads, self.d_rel),
                    q_scale=self.q_scale, r_scale=self.r_scale, tau=tau, q_thr=self.qk_clip, r_thr=self.rel_clip,
                    k_thr=self.qk_clip, eps=self.eps, cache=kv_cache, slots=attn_md[layer_name].slot_mapping,
                )
                k = v = None
            elif _USE_TRITON and q.device.type == "xpu":
                q = rows_op(q.view(num_tokens, self.num_heads, self.head_dim), mode=0,
                            threshold=self.qk_clip, scale=self.q_scale, tau=tau)
                k = rows_op(k.view(num_tokens, self.num_kv_heads, self.head_dim), mode=0,
                            threshold=self.qk_clip)
                v = rows_op(v.view(num_tokens, self.num_kv_heads, self.head_dim), mode=1,
                            eps=self.eps)
                r = rows_op(r.view(num_tokens, self.num_heads, self.d_rel), mode=0,
                            threshold=self.rel_clip, scale=self.r_scale, tau=tau)
            else:
                q = _l2_clip(q.view(num_tokens, self.num_heads, self.head_dim), self.qk_clip, self.q_scale)
                k = _l2_clip(k.view(num_tokens, self.num_kv_heads, self.head_dim), self.qk_clip)
                v = _v_norm(v.view(num_tokens, self.num_kv_heads, self.head_dim), self.eps)
                r = _l2_clip(r.view(num_tokens, self.num_heads, self.d_rel), self.rel_clip, self.r_scale)
                if tau is not None:
                    q = (q.float() * tau[:, None, None]).to(torch.bfloat16)
                    r = (r.float() * tau[:, None, None]).to(torch.bfloat16)

        if _kv_sim.MODE[0] != "off":
            k, v = _kv_sim.simulate(k, v)
        if not has_cache:
            # Profile/dummy run: no paged cache is bound; run dense attention.
            return self.wo_apply(
                _dense_attention(
                    q,
                    k,
                    v,
                    r,
                    gate,
                    sink,
                    self.rel_logits_proj,
                    positions,
                    self.rel_extent,
                    self.sliding_window,
                    self.scaling,
                )
            )
        md = attn_md[layer_name]
        with _ph("a.core"):
          attn = _paged_attention(
            q,
            k,
            v,
            r,
            gate,
            sink,
            self._rel_f32 if self._rel_f32 is not None else self.rel_logits_proj,
            kv_cache,
            md,
            positions,
            self.rel_extent,
            self.sliding_window,
            self.scaling,
            skip_kv_write=fused,
        )
        with _ph("a.wo"):
            y = self.wo_apply(attn)
        return y if out is None else out.copy_(y)
