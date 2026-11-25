# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini (looped) for Intel XPU.

Mirrors the NVIDIA implementation's structure and weight layout but runs
entirely on torch + Triton-XPU: custom paged attention (rel bias + sink +
gate), paged short-conv state, expert-sharded MoE with the SITU activation, and
the looped cross-pass message controllers.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Callable

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from ..processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from ..quant_config import Inkling2QuantConfig
from .attention import InklingXPUAttention
from .loop import LOOP_STREAMS, LoopControllers, attn_res_mix
from .triton_loop import ctrl_blend_norm_op, mix_op

import os as _os_early

_FAST = _os_early.environ.get("INKLING_XPU_TORCH_LOOP", "0") != "1"
from .moe import Inkling2XPUMoE, expert_formats
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2XPUShortConv
from .sconv_state import InklingConvState

logger = init_logger(__name__)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    y = y * weight.float()
    return y if out_dtype is None else y.to(out_dtype)


class Inkling2XPURMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks per-input norm weights."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs, hidden_size), requires_grad=False)
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return _rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2XPUAttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32, pre-scaled)."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros((1, hidden) if i in local_ids else (3, hidden), dtype=torch.float32),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


import os as _os_mod
import time as _time

_os_env = _os_mod.environ

from . import prof as _prof
from .prof import PROF, ph as _ph

_PROF_T = _prof.TIMES
_PROF_N = [0]


DEBUG_DUMP: list | None = None  # parity debugging: per-visit tensors


class Inkling2XPUDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        prefix: str,
        num_passes: int = 1,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        )
        self.conv_states = tuple(
            InklingConvState(
                num_kv_heads=num_kv_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state" + (f".pass{p}" if p else ""),
                replicate_hidden=True,
            )
            for p in range(num_passes)
        )
        self.conv_state = self.conv_states[0]
        self.attn = InklingXPUAttention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=self.conv_states[1:],
        )
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2XPURMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2XPURMSNorm(config.hidden_size, eps)
        self.mlp = Inkling2XPUMoE(config, expert_formats(get_current_vllm_config()).get(layer_id))
        self.attn_sconv, self.mlp_sconv = (
            Inkling2XPUShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )
        if layer_id in config.loop_core_layers:
            self.controllers = LoopControllers(config.hidden_size, config.loop_config["gate_rank"])  # type: ignore[index]
        else:
            self.controllers = None

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], out=out, pass_idx=pass_idx)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            out,
            pass_idx,
        )


class Inkling2XPUModel(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        assert self.normalize_values
        self.embed_tokens = VocabParallelEmbedding(config.padded_vocab_size, config.hidden_size)
        self.embed_norm = Inkling2XPURMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            self._make_layer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2XPUAttnRes(config)
        self.norm = Inkling2XPURMSNorm(config.hidden_size, config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2XPUDecoderLayer:
        return Inkling2XPUDecoderLayer(config, layer_id, prefix)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self,
        blocks: Sequence[torch.Tensor],
        partial: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2XPURMSNorm,
        addend: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Mix `[*blocks, partial (+ addend)]` per query row and apply `norm`."""
        last = partial if addend is None else partial + addend.float()
        mixed = attn_res_mix(blocks, last, query)
        return [
            norm(mixed[j], j if norm.weight.shape[0] > 1 else 0)
            for j in range(mixed.shape[0])
        ]

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        floor = self.config.log_scaling_n_floor
        if floor is None:
            return None
        tau = 1.0 + self.config.log_scaling_alpha * torch.log(
            ((positions.float() + 1) / floor).clamp(min=1.0)
        )
        return tau  # == 1.0 below log_scaling_n_floor; no host sync (graph-capture safe)

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens = inputs_embeds.shape[0]
        tau = self._log_scaling_tau(positions)
        aux_hidden_states: list[torch.Tensor] = []
        blocks: list[torch.Tensor] = []
        stream = inputs_embeds.float()
        for i, layer in enumerate(self.layers):
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(stream.clone().to(torch.bfloat16))
            if i == 0:
                inputs = [layer.attn_norm(stream, j) for j in range(layer.num_attn_inputs)]
            else:
                query = self.attn_res.attn_queries[i - 1]
                inputs = self._mix(blocks, stream, query, layer.attn_norm)
            attn_partial = layer.attention(positions, inputs, tau)
            attn_out = layer.attn_sconv(
                tensor_model_parallel_all_reduce(attn_partial), positions
            )
            first = i % self.block_size == 0
            mlp_query = self.attn_res.mlp_queries[i]
            if first:
                mlp_in = self._mix([*blocks, stream], attn_out, mlp_query, layer.mlp_norm)
            else:
                mlp_in = self._mix(blocks, stream, mlp_query, layer.mlp_norm, attn_out)
            residual = None if first else stream
            if first:
                blocks.append(stream)
            moe_out = layer.mlp(mlp_in[0])
            stream = layer.mlp_sconv(moe_out, positions, addend=attn_out, residual=residual)
            stream = stream.float()
        out = self._mix(blocks, stream, self.attn_res.output_query, self.norm)[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2XPULoopModel(Inkling2XPUModel):
    """The looped variant: the core layers run `num_passes` times."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        hidden = config.hidden_size
        shape = (2 * len(self.loop_core), hidden)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(len(self.loop_core), len(LOOP_STREAMS), hidden, dtype=torch.float32),
            persistent=False,
        )

    def _make_layer(self, config, layer_id, quant_config, prefix):
        num_passes = (
            config.num_loop_passes if layer_id in config.loop_core_layers else 1
        )
        return Inkling2XPUDecoderLayer(config, layer_id, prefix, num_passes=num_passes)

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.config.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(-1, 2, self.config.hidden_size)
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.config.hidden_size**-0.5)

    def _messages(self, layer_idx: int, bank: Sequence[torch.Tensor]) -> torch.Tensor:
        query = self.message_queries[layer_idx - self.loop_core.start]
        if _FAST and bank[-1].device.type == "xpu":
            return mix_op(list(bank[:-1]), bank[-1], query)  # [4, T, D] fp32
        return attn_res_mix(bank[:-1], bank[-1], query)  # [4, T, D] fp32

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stream: torch.Tensor,
        blocks: list[torch.Tensor],
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        bank: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layer = self.layers[i]
        messages = None
        if bank is not None:
            messages = self._messages(i, bank)
            assert layer.controllers is not None
        fast = _FAST and stream.device.type == "xpu"
        with _ph("pre_attn"):
          if fast and i > 0:
            query = self.attn_res.attn_queries[i - 1]
            if messages is None:
                out = mix_op(blocks, stream, query, norm_weight=layer.attn_norm.weight,
                             norm_eps=layer.attn_norm.eps)
                attn_inputs = [out[j] for j in range(out.shape[0])]
            else:
                mixed = mix_op(blocks, stream, query)  # fp32 [nq, T, D]
                dest = mixed[:1].expand(3, *mixed.shape[1:]) if layer.is_local else mixed
                cs = layer.controllers
                out = ctrl_blend_norm_op(
                    dest, messages[:3], cs.down_all[:3], cs.gain_all[:3], cs.forget_all[:3],
                    layer.attn_norm.weight, layer.mlp_norm.eps, layer.attn_norm.eps,
                )
                attn_inputs = [out[j] for j in range(3)]
          else:
            # attention inputs (mixes of the block bank, or the raw stream at layer 0)
            if i == 0:
                mixed = [stream]
            else:
                query = self.attn_res.attn_queries[i - 1]
                mixed = attn_res_mix(blocks, stream, query)  # [nq, T, D]
            if messages is not None:
                assert layer.controllers is not None
                eps = layer.mlp_norm.eps
                if layer.is_local:
                    # Each stream's blend is its own SWA input, all normed alike.
                    attn_inputs = [
                        layer.attn_norm(
                            layer.controllers.blend(LOOP_STREAMS[j], mixed[0], messages[j], eps),
                            0,
                        )
                        for j in range(3)
                    ]
                else:
                    attn_inputs = [
                        layer.attn_norm(
                            layer.controllers.blend(LOOP_STREAMS[j], mixed[j], messages[j], eps), j
                        )
                        for j in range(3)
                    ]
            elif i == 0:
                attn_inputs = [layer.attn_norm(stream, j) for j in range(layer.num_attn_inputs)]
            else:
                attn_inputs = [layer.attn_norm(mixed[j], j) for j in range(mixed.shape[0])]

        with _ph("attention"):
            attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)
        with _ph("attn_allreduce"):
            attn_red = tensor_model_parallel_all_reduce(attn_partial)
        with _ph("attn_sconv"):
            # fast path: the conv emits the bf16-rounded value already widened to fp32 (== the old `.float()`), so the
            # later `stream + attn_out.float()` / mix input / mlp_sconv addend consume it without a cast kernel
            attn_out = layer.attn_sconv(attn_red, positions, out_dtype=torch.float32 if fast else None)
        first = i % self.block_size == 0
        with _ph("pre_mlp"):
          if fast:
            mlp_query = self.attn_res.mlp_queries[i]
            if first:
                cur = attn_out
                srcs = [*blocks, stream]
            else:
                cur = stream + attn_out.float()
                srcs = blocks
            if messages is None:
                # `first` visits: `cur` used to be bf16 (=> stacked kernel); keep that kernel so the numerics stay bit-identical
                mlp_in = [mix_op(srcs, cur, mlp_query, norm_weight=layer.mlp_norm.weight,
                                 norm_eps=layer.mlp_norm.eps, allow_mp=not first)[0]]
            else:
                mixed = mix_op(srcs, cur, mlp_query, allow_mp=not first)  # fp32 [1, T, D]
                cs = layer.controllers
                mlp_in = [ctrl_blend_norm_op(
                    mixed, messages[3:4], cs.down_all[3:], cs.gain_all[3:], cs.forget_all[3:],
                    layer.mlp_norm.weight, layer.mlp_norm.eps, layer.mlp_norm.eps,
                )[0]]
          else:
            mlp_query = self.attn_res.mlp_queries[i]
            if first:
                mlp_in = attn_res_mix([*blocks, stream], attn_out, mlp_query)
            else:
                mlp_in = attn_res_mix(blocks, stream + attn_out.float(), mlp_query)
            if messages is not None:
                assert layer.controllers is not None
                mlp_in = layer.controllers.blend(
                    "mlp", mlp_in[0], messages[3], layer.mlp_norm.eps
                )
                mlp_in = [layer.mlp_norm(mlp_in)]
            else:
                mlp_in = [layer.mlp_norm(mlp_in[0])]

        residual = None if first else stream
        if first:
            blocks = [*blocks, stream]
        with _ph("moe"):
            moe_out = layer.mlp(mlp_in[0])
        with _ph("mlp_sconv"):
            out = layer.mlp_sconv(moe_out, positions, addend=attn_out, residual=residual,
                                  out_dtype=torch.float32 if fast else None)
        if DEBUG_DUMP is not None:
            DEBUG_DUMP.append(
                {"i": i, "pass": pass_idx, "attn_partial": attn_partial.float().cpu(), "attn_out": attn_out.float().cpu(),
                 "mlp_in": mlp_in[0].float().cpu(), "moe": moe_out.float().cpu(),
                 "stream": out.float().cpu()}
            )
        return out.float(), blocks

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return _prof.tprof_call("target", inputs_embeds.shape[0], self._forward_impl, positions, inputs_embeds)

    def _forward_impl(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        _tw = _time.perf_counter() if PROF else 0.0
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        tau = self._log_scaling_tau(positions)
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]

        aux_hidden_states: list[torch.Tensor] = []
        stream = inputs_embeds.float()
        blocks: list[torch.Tensor] = []
        prelude_blocks: list[torch.Tensor] = []
        prelude_stream: torch.Tensor | None = None
        bank: Sequence[torch.Tensor] | None = None
        for visit, (i, pass_idx) in enumerate(visits):
            if i == core.start and pass_idx == 0:
                prelude_blocks = list(blocks)
                prelude_stream = stream.clone()
            elif i == core.start:
                assert prelude_stream is not None
                bank = [*blocks[len(prelude_blocks) + 1 :], stream.clone()]
                blocks = list(prelude_blocks)
                stream = prelude_stream.clone()
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stream.clone().to(torch.bfloat16))
            stream, blocks = self._run_layer(
                i,
                pass_idx,
                stream,
                blocks,
                positions,
                tau,
                bank if pass_idx > 0 else None,
            )
        out = self._mix(blocks, stream, self.attn_res.output_query, self.norm)[0]
        if PROF:
            torch.xpu.synchronize()
            _PROF_T["forward_wall"] = _time.perf_counter() - _tw
            _PROF_N[0] += 1
            if inputs_embeds.shape[0] > 64 or _PROF_N[0] % 16 == 0:
                print(f"PROF T={inputs_embeds.shape[0]} (ms):", {k: round(v * 1e3) for k, v in _PROF_T.items()}, flush=True)
            _PROF_T.clear()
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


_HEAD_ROW_CHUNK = 16  # keep every head GEMM below the oneMKL M=26 kernel switch


class _XPULogitsProcessor(LogitsProcessor):
    """fp32 LM head without materialising an fp32 copy of the weight each step.

    vLLM's generic fp32-head path on non-CUDA/ROCm devices casts the whole `[V, D]` weight to
    fp32 per call (8 ms/token for a 201K vocab). `torch.bmm(out_dtype=float32)` accumulates the
    bf16 operands straight into fp32 and only reads the bf16 weight once.
    """

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        if (
            self.head_dtype == torch.float32
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.device.type == "xpu"
            and lm_head.weight.dtype == torch.bfloat16
        ):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            # OneMKL switches GEMM kernels at M=26 on this stack (torch 2.15.0+xpu,
            # SYCL 2026.1.2): rows<=25 are bit-identical to the per-row bmm, rows
            # 26-32 diverge (measured on this stack). Chunk at _HEAD_ROW_CHUNK so
            # every GEMM stays in the exact regime; the head reads the bf16 weight
            # once per chunk instead of once per row.
            # single GPU, full 201K vocab: rows=4 6.21 -> 1.59 ms,
            # rows=16 24.76 -> 1.68, rows=32 49.49 -> 3.37. Verified rows=1..32,
            # 20 repeats, 3 fresh processes, row-position invariance true.
            # The per-row loop was adopted for the batched-matmul mode-B flip;
            # that defect is batch>1 bmm, not these batch=1 GEMMs. Old path kept
            # one release: INKLING2_HEAD_PER_ROW=1 (kept for one release).
            # Re-run the row sweep on any torch/SYCL/driver upgrade.
            # Do not pre-transpose to a contiguous [H,V] copy (changes the last
            # bits, max 2.15e-6) and do not materialize a repeated weight (696 ms).
            if os.environ.get("INKLING2_HEAD_PER_ROW", "0") == "1":
                wt = lm_head.weight.t().unsqueeze(0)
                logits = torch.cat([
                    torch.bmm(flat[i:i + 1].unsqueeze(0), wt, out_dtype=torch.float32)[0]
                    for i in range(flat.shape[0])
                ])
            else:
                logits = torch.cat([
                    torch.mm(flat[i:i + _HEAD_ROW_CHUNK],
                             lm_head.weight.t(), out_dtype=torch.float32)
                    for i in range(0, flat.shape[0], _HEAD_ROW_CHUNK)
                ])
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(torch.float32)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return super()._apply_head(lm_head, hidden_states, embedding_bias)



_IMAGE_PAD_ID = 200054  # image placeholder token id; must match processing.IMAGE_PAD_ID


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2XPUForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsEagle3):
    """Inkling 2 Mini for XPU (text and images; vision tower ported from the reference XPU implementation)."""

    # Keep raw input_ids through the runner: the model embeds internally (inside
    # the captured decode graph, exactly like text-only boots) and overlays the
    # runner-merged multimodal embeddings itself. Without this, mm-active boots
    # embed outside the graph and every decode forward diverges bit-wise.
    requires_raw_input_tokens = True

    packed_modules_mapping: dict[str, list[str]] = {}

    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        self.config = config
        self.text_config = text_config = config.text_config
        mm = vllm_config.model_config.multimodal_config
        self.visual = None
        _force_visual = os.environ.get("VLLM_INKLING2_FORCE_VISUAL") == "1"
        if (getattr(config, "vision_config", None) is not None
                and (_force_visual or (mm is not None and mm.get_limit_per_prompt("image")))):
            from .vision import Inkling2VisionEncoder

            self.visual = Inkling2VisionEncoder(config.vision_config, text_config.hidden_size)
        model_cls = (
            Inkling2XPULoopModel if text_config.loop_config is not None else Inkling2XPUModel
        )
        self.model = model_cls(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(text_config.padded_vocab_size, text_config.hidden_size)
        self.logits_processor = _XPULogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )
        self.ring_prefix_worker = None
        if os.environ.get("INK2_RING_PREFIX_CACHE", "0") == "1":
            if not vllm_config.cache_config.enable_prefix_caching:
                raise ValueError(
                    "INK2_RING_PREFIX_CACHE requires --enable-prefix-caching"
                )
            if vllm_config.scheduler_config.max_num_seqs != 1:
                raise ValueError(
                    "INK2_RING_PREFIX_CACHE requires --max-num-seqs 1"
                )
            from .ring_prefix_worker import RingPrefixWorker

            self.ring_prefix_worker = RingPrefixWorker(
                vllm_config.compilation_config.static_forward_context,
                vllm_config,
            )

    def get_language_model(self) -> nn.Module:
        return self

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches, grid_thw = kwargs.get("vision_pixels"), kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor) and isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        return self.visual(patches, grid_thw).split(grid_thw.prod(-1).tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(inputs_embeds=inputs_embeds,
                                            multimodal_embeddings=multimodal_embeddings,
                                            is_multimodal=is_multimodal)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        worker = self.ring_prefix_worker
        orig_embeds = inputs_embeds
        if worker is not None:
            early = worker.entry(input_ids, positions, orig_embeds)
            if early is not None:
                return early
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
            output = self.model(positions, inputs_embeds)
            if worker is not None:
                worker.exit(input_ids, orig_embeds, output)
            return output
        # requires_raw_input_tokens path (mm-active boots): the runner passes the
        # merged embeds alongside input_ids. Re-embed the tokens here (identical
        # ops to text-only boots, inside the graph) and overlay the encoder
        # outputs at the image rows only. Text rows of the runner embeds are
        # bit-identical to this re-embed, so the overlay is inert for them.
        assert input_ids is not None
        embeds = self.model.embed_input_ids(input_ids)
        # While the stream is capturing, skip the overlay entirely: host syncs
        # (.any()) are illegal during capture (level_zero error 44), and the
        # captured decode graph must stay byte-identical to text-only boots.
        # Decode rows never carry image embeddings (image tokens live in the
        # prompt KV cache), so this is semantically exact; eager prefill steps
        # below apply the overlay for real image rows.
        if not torch.cuda.is_current_stream_capturing():
            mask = input_ids == _IMAGE_PAD_ID  # image placeholder rows
            if bool(mask.any()):
                overlay = inputs_embeds[: embeds.shape[0]].to(embeds.dtype)
                cond = mask.unsqueeze(-1).expand_as(embeds).contiguous()
                embeds = torch.where(cond, overlay, embeds)
        output = self.model(positions, embeds)
        if worker is not None:
            worker.exit(input_ids, orig_embeds, output)
        return output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_items: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_items.append((name[len("model.visual."):], weight))
                continue  # vision tower (loaded via self.visual below); mtp.safetensors skipped
            if not name.startswith("model."):
                continue  # mtp.safetensors (draft weights, loaded by the speculator)
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name)
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype and not getattr(param, "byte_param", False):
                raise ValueError(
                    f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}"
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        if self.visual is not None and visual_items:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_items)}
        # The fused qkvr weight carries a trailing alignment-pad shard that the
        # checkpoint never provides: fill it with zeros so rows are defined and
        # the loader's per-shard bookkeeping sees it as loaded.
        for mod_name, mod in self.named_modules():
            if isinstance(mod, InklingXPUAttention) and len(mod.split_sizes) > 6:
                qkvr = mod.qkvr
                zeros = torch.zeros(
                    qkvr.output_sizes[6], qkvr.input_size, dtype=qkvr.weight.dtype
                )
                qkvr.weight_loader(qkvr.weight, zeros, 6)
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded


__all__ = [
    "Inkling2XPUForConditionalGeneration",
    "Inkling2XPULoopModel",
    "Inkling2XPUModel",
]
