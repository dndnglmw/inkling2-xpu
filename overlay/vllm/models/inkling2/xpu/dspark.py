# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (`mtp.safetensors`) for Intel XPU.

Same architecture / numerics as `nvidia/dspark.py` (an anchor plus mask block run
through a small dense transformer over DFlare-mixed target-feature K/V, then a
recurrent `rnn_hidden` Markov head), with the CUDA-only pieces replaced:
flashinfer RMSNorm / fused all-reduce -> torch, libdevice Triton helper ops ->
torch (rounding points kept), `block_decode_attention` -> a paged torch block
attention (first version; a Triton kernel can replace it), FlashAttention KV
cache owner -> `InklingXPUDraftKVCache` using the same cache layout and metadata
builder as the target's XPU attention.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheLayerConfig

from ..configs import Inkling2DSparkConfig
from ..speculator_config import Inkling2DSparkSpeculatorConfig
from .attention import InklingXPUAttentionBackend, _kv_cache_layer_config
from .triton_block_attention import triton_block_attention
from .prof import PROF, TIMES as _TIMES, ph as _ph, tprof_call as _tprof_call
from .triton_prep import kv_write

_TORCH_BLOCK_ATTN = __import__("os").environ.get("INKLING_XPU_TORCH_BLOCK_ATTN", "0") == "1"
_DRAFT_GLOBAL_WINDOW = int(__import__("os").environ.get("INKLING_XPU_DRAFT_GLOBAL_WINDOW", "0"))


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor to bf16 precision (kept in fp32)."""
    return x.to(torch.bfloat16).to(torch.float32)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            # residual += x (rounded to bf16); x = norm(unrounded sum)
            s = residual.float() + x.float()
            residual.copy_(s.to(residual.dtype))
        else:
            s = x.float()
        y = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def all_reduce_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, norm: _RMSNorm) -> torch.Tensor:
    """`residual += all_reduce(x); x = norm(residual)`."""
    return norm(tensor_model_parallel_all_reduce(x), residual)


def mix_rmsnorm(targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of the `alpha`-weighted sum of `targets[:, i]` (fp32 math), `(rows, taps, hidden)`."""
    mixed = (targets.float() * alpha.float()[None, :, None]).sum(1)
    y = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(targets.dtype)


def _rope_neox(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """neox RoPE on `(T, H, D)` with fp32 `[cos | sin]` table; fp32 math, bf16 out."""
    d = x.shape[-1]
    half = d // 2
    cs = cos_sin_cache[positions.long()]
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)


def _head_rmsnorm_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm rounded to bf16 (as the reference)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(torch.bfloat16)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor:
    x = (a.float() + b.float()).to(torch.bfloat16).float()
    return (torch.tanh(x / softcap) * softcap).to(torch.bfloat16)


def markov_gate(raw: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent gating of the fp32 `rnn_proj` output (rounded like the bf16 reference ops)."""
    r = state.shape[-1]
    gate = _bf16(raw[:, :r])
    cand = _bf16(raw[:, r : 2 * r])
    out = _bf16(raw[:, 2 * r :])
    gate = _bf16(torch.sigmoid(gate))
    cand = _bf16(torch.tanh(cand))
    st = state.float()
    keep = _bf16(gate * st)
    update = _bf16(_bf16(1.0 - gate) * cand)
    return (keep + update).to(torch.bfloat16), torch.tanh(out).to(torch.bfloat16)


class InklingXPUDraftKVCache(nn.Module, AttentionLayerBase):
    """Owns one draft layer's paged KV cache (`[nb, Hkv, bs, 2D]`, K|V adjacent), same
    layout/spec/metadata as the target's XPU attention layers."""

    def __init__(self, prefix: str, num_kv_heads: int, head_dim: int, sliding_window: int | None) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_name = prefix
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        static = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in static:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static[prefix] = self

    def get_attn_backend(self):
        return InklingXPUAttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window, dtype=torch.bfloat16
        )


def block_attention(
    q: torch.Tensor,  # [T, H, D] (T = reqs * block), roped + normed
    kv_cache: torch.Tensor,  # [nb, Hkv, bs, 2D]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block: int,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Non-causal attention of each request's `block` query rows over its last `window`
    keys (all keys when `window == 0`), GQA; returns `[T, H*D]`."""
    t, h, d = q.shape
    hkv = kv_cache.shape[1]
    groups = h // hkv
    bs = kv_cache.shape[2]
    out = torch.zeros(t, h, d, device=q.device, dtype=torch.bfloat16)
    lens = seq_lens.tolist()
    for r in range(min(t // block, len(lens))):
        length = int(lens[r])
        start = max(0, length - window) if window else 0
        pos = torch.arange(start, length, device=q.device)
        lblk = (pos // bs).clamp(max=block_table.shape[1] - 1)
        blocks = block_table[r, lblk].long().clamp(min=0)
        kv = kv_cache[blocks, :, pos % bs, :]  # [S, Hkv, 2D]
        k = kv[..., :d].float().repeat_interleave(groups, dim=1)
        v = kv[..., d:].float().repeat_interleave(groups, dim=1)
        qr = q[r * block : (r + 1) * block].float()
        scores = torch.einsum("q h d, s h d -> h q s", qr, k) * scale
        probs = torch.softmax(scores, dim=-1)
        out[r * block : (r + 1) * block] = torch.einsum("h q s, s h d -> q h d", probs, v).to(torch.bfloat16)
    return out.reshape(t, h * d)


class Inkling2DraftSelfAttention(nn.Module):
    """Bidirectional block attention over context K/V (from target features) and the block
    itself; local layers see `sliding_window_size` positions before the block start."""

    def __init__(self, config: Inkling2DSparkConfig, is_local: bool, prefix: str) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.block = config.draft_block_size
        self.is_local = is_local
        self.qkv_proj = QKVParallelLinear(
            hidden, self.head_dim, config.num_attention_heads, config.num_key_value_heads,
            bias=False, prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden, [config.num_key_value_heads * self.head_dim] * 2, bias=False, prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, hidden, bias=False, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm_ctx = _RMSNorm(self.head_dim, self.eps)
        self.attn = InklingXPUDraftKVCache(
            f"{prefix}.attn", self.num_kv_heads, self.head_dim,
            config.sliding_window_size if is_local else None,
        )
        # Keys a block sees: itself and, if local, the window before it.
        # global draft layer: full context by default; INKLING_XPU_DRAFT_GLOBAL_WINDOW=N bounds it to the last N keys
        # (decode-speed experiment: cuts the draft's context-proportional sweep, may lower acceptance).
        self.window = config.sliding_window_size + self.block if is_local else (_DRAFT_GLOBAL_WINDOW + self.block if _DRAFT_GLOBAL_WINDOW else 0)

    def _metadata(self):
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.attn.kv_cache.numel() == 0:
            return None
        return attn_metadata[self.attn.layer_name]

    def forward(self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(x, self.qkv_proj.weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        t = q.shape[0]
        q = _head_rmsnorm_bf16(q.reshape(t, self.num_heads, self.head_dim), self.q_norm.weight, self.eps)
        k = _head_rmsnorm_bf16(k.reshape(t, self.num_kv_heads, self.head_dim), self.k_norm.weight, self.eps)
        q = _rope_neox(q, positions, cos_sin_cache)
        k = _rope_neox(k, positions, cos_sin_cache)
        v = v.reshape(t, self.num_kv_heads, self.head_dim)
        out = q.new_zeros((t, self.q_size))
        md = self._metadata()
        if md is not None:
            kv_write(k.contiguous(), v.contiguous(), self.attn.kv_cache, md.slot_mapping[:t])
            num_reqs = min(t // self.block, md.seq_lens.shape[0])
            n = num_reqs * self.block
            attn_fn = block_attention if _TORCH_BLOCK_ATTN else triton_block_attention
            out[:n] = attn_fn(
                q[:n], self.attn.kv_cache, md.block_table[:num_reqs], md.seq_lens[:num_reqs],
                self.block, self.window, self.head_dim**-0.5,
            )
        return F.linear(out, self.o_proj.weight)

    def store_context_kv(
        self, context: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        k, v = F.linear(context, self.kv_proj_ctx.weight).split([self.kv_size, self.kv_size], -1)
        k = _head_rmsnorm_bf16(k.reshape(-1, self.num_kv_heads, self.head_dim), self.k_norm_ctx.weight, self.eps)
        k = _rope_neox(k, positions, cos_sin_cache)
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            kv_write(k.contiguous(), v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous(),
                     self.attn.kv_cache, slot_mapping)


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = F.linear(x, self.gate_up_proj.weight)
        n = gu.shape[-1] // 2
        gate, up = gu[..., :n].float(), gu[..., n:].float()
        act = (F.silu(gate) * up).to(x.dtype)
        return F.linear(act, self.down_proj.weight)


class Inkling2DraftLayer(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, is_local: bool, prefix: str) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)
        self.dflare_norm = _RMSNorm(hidden, eps)
        self.self_attn = Inkling2DraftSelfAttention(config, is_local, f"{prefix}.self_attn")
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(self, positions, x, residual, cos_sin_cache):
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x = all_reduce_add_rmsnorm(x, residual, self.input_layernorm)
        x = self.self_attn(positions, x, cos_sin_cache)
        x = all_reduce_add_rmsnorm(x, residual, self.post_attention_layernorm)
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits (see nvidia/dspark.py)."""

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size, rank, _weight=torch.empty(config.padded_vocab_size, rank), _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold `hidden_out` into the (TP-local) LM head rows in fp64 (on CPU: XPU has no fp64)."""
        projection = self.hidden_out.weight.double().cpu()
        bias = self.hidden_out.bias.double().cpu()
        num_rows = lm_head_weight.shape[0]
        if self.fold_weight is None or self.fold_bias is None:
            self.fold_weight = lm_head_weight.new_empty((num_rows, self.markov_rank))
            self.fold_bias = lm_head_weight.new_empty((num_rows,))
        for start in range(0, num_rows, 8192):
            rows = slice(start, start + 8192)
            head = lm_head_weight[rows].double().cpu()
            self.fold_weight[rows].copy_((head @ projection).to(lm_head_weight.dtype))
            self.fold_bias[rows].copy_((head @ bias).to(lm_head_weight.dtype))

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        assert self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        out = self.rnn_bias + inputs.flatten(0, 1).float() @ weight.float().T
        return out.view(hidden.shape[0], num_reqs, -1)

    def step(self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor):
        assert self.fold_weight is not None and self.fold_bias is not None
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
        raw = context + inputs.float() @ weight.float().T
        state, latent = markov_gate(raw.contiguous(), state)
        return state, F.linear(latent, self.fold_weight, self.fold_bias)


class Inkling2DSparkDecoder(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, max_model_len: int, prefix: str) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(config, i in config.local_layer_ids, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids)), requires_grad=False
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.register_buffer("fuse_alpha", torch.empty_like(self.fuse_weight))
        self.cos_sin_cache = get_rope(
            config.head_dim, max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta}, dtype=torch.float32,
        ).cos_sin_cache
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.config.draft_block_size
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous())).to(x.dtype)
        return x.view(-1, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin_cache)
        assert residual is not None
        return all_reduce_add_rmsnorm(x, residual, self.norm)


_DRAFT_N = [0]


class Inkling2XPUDSparkModel(nn.Module):
    """DSpark draft for Inkling2 on XPU; plugged in by `Inkling2DSparkSpeculator`."""

    speculator_config_class = Inkling2DSparkSpeculatorConfig
    allow_patterns_overrides = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        max_depth = getattr(config, "max_draft_block_size", config.draft_block_size) - 1
        if not 1 <= num_spec <= max_depth:
            raise ValueError(
                f"Inkling2 DSpark on XPU takes num_speculative_tokens in [1, {max_depth}] "
                f"(the checkpoint's block size - 1), got {num_spec}"
            )
        # Shorter verified depth: the draft block (anchor + masks) is sized to depth + 1 rows, which is exactly what
        # the speculator feeds (1 + num_speculative_tokens rows per request) and what the scheduler allocates KV
        # slots for. Verify cost then scales with the depth (a depth-3 step is ~30 ms vs ~43 ms at depth 7).
        config.max_draft_block_size = max_depth + 1
        config.draft_block_size = num_spec + 1
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(f"set --max-model-len to at most {config.max_position_embeddings}")
        self.config = config
        self.model = Inkling2DSparkDecoder(config, vllm_config.model_config.max_model_len, f"{prefix}model")
        self.lm_head: nn.Module | None = None

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        assert self.lm_head is not None and self.lm_head.weight.dtype == torch.bfloat16
        self.model.embed_norm = target_model.model.embed_norm
        head = self.model.markov_head
        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        assert self.lm_head is not None
        head = self.model.markov_head
        assert head.rnn_bias is not None
        head.fold(self.lm_head.weight)
        head.rnn_bias.copy_(head.rnn_proj.bias)
        self.model.fuse_alpha.copy_(torch.softmax(self.model.fuse_weight.float(), dim=-1))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def precompute_and_store_context_kv(
        self, context_states: torch.Tensor, context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        with _ph("draft.ctx_kv"):
          for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context, context_positions, self.model.cos_sin_cache,
                None if context_slot_mappings is None else context_slot_mappings[i],
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds=None) -> torch.Tensor:
        with _ph("draft.forward"):
            out = _tprof_call("draft", input_ids.shape[0], self.model, input_ids, positions)
        _DRAFT_N[0] += 1
        if PROF and _DRAFT_N[0] % 16 == 0:
            print("DRAFT PROF (ms/step):", {k: round(v / 16 * 1e3, 1) for k, v in _TIMES.items() if k.startswith("draft")}, flush=True)
            for k in [k for k in _TIMES if k.startswith("draft")]:
                _TIMES.pop(k)
        return out

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        with _ph("draft.base_logits"):
            return F.linear(hidden_states, self.lm_head.weight)

    def markov_step(self, state, prev_tokens, context):
        with _ph("draft.markov_step"):
            return self.model.markov_head.step(state, prev_tokens, context)

    def markov_logits(self, base_logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def markov_argmax(self, base_logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        with _ph("draft.markov_argmax"):
            logits = self.markov_logits(base_logits, bias).float()
            return torch.nan_to_num(logits, nan=float("-inf")).argmax(-1)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype != param.dtype:
                raise ValueError(f"{name}: expected {param.dtype}, got {weight.dtype}")
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded
