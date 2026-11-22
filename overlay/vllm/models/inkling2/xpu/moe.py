# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE for Intel XPU: fp32 router, expert-sharded experts, SITU.

Routing (validated against llama.cpp and vLLM NVIDIA):
    scores = sigmoid(x @ W_gate.T)                (fp32)
    sel    = scores + e_score_correction_bias    (selection only)
    idx    = topk(sel, k)                        (weights gathered from scores)
    w      = scores[idx] / sum(scores[idx]) * route_scale

The checkpoint stores `w13` with *interleaved* (gate_i, up_i) rows. We permute
the fused rows once at load into the packed [all gates; all ups] layout so the
SITU activation (`situ_and_mul`, beta=gate softcap, linear_beta=up softcap) can
use contiguous halves:

    gate = beta * tanh(gate / beta) * sigmoid(gate)
    up   = linear_beta * tanh(up / linear_beta)
    out  = gate * up
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.utils import set_weight_attrs

from .triton_moe import triton_moe_small
from .triton_router import router_topk
from .triton_moe_grouped import triton_moe_grouped_q
from .triton_moe_q import FMT_MXFP8, FMT_NVFP4, dequant_expert, triton_moe_small_q

_SMALL_T = 8  # token count up to which the per-pair streaming kernel is used
_DUMP_DIR = os.environ.get("INKLING_MOE_DUMP")  # debug: save routing indices of the first 48 large-T visits
_DUMP_N = [0]
_DUMP_ON = [False]  # set True only around the real request (see agents/routing/capture.py)
_TORCH_ROUTER = os.environ.get("INKLING_XPU_TORCH_ROUTER", "0") == "1"
_TORCH_MOE = os.environ.get("INKLING_XPU_TORCH_MOE", "0") == "1"


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Torch fallback for `SituAndMul` on packed [gate | up] halves."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


try:  # vllm_xpu_kernels ships situ_and_mul on XPU
    _SITU_OP = torch.ops._C.situ_and_mul
except (AttributeError, RuntimeError):  # pragma: no cover - CPU dev
    _SITU_OP = None


class Inkling2XPUGate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": lambda p, w: p.data.copy_(w)})


import functools


def expert_formats(vllm_config) -> dict[int, str]:
    return _expert_formats(str(vllm_config.model_config.model))


@functools.lru_cache(maxsize=4)
def _expert_formats(model_dir: str) -> dict[int, str]:
    """layer id -> 'nvfp4' | 'mxfp8' from `hf_quant_config.json` next to the model."""
    import json
    import re

    path = os.path.join(model_dir, "inkling_expert_quant.json")
    if not os.path.exists(path):
        path = os.path.join(model_dir, "hf_quant_config.json")
    if not os.path.exists(path):
        return {}
    out: dict[int, str] = {}
    for group in json.load(open(path)).get("model", {}).get("parameter_groups", []):
        fmt = group.get("compute_dtype")
        if fmt not in ("nvfp4", "mxfp8"):
            continue
        for name in group.get("tensors", []):
            m = re.search(r"\.layers\.(\d+)\.mlp\.experts\.w13_weight$", name)
            if m:
                out[int(m.group(1))] = fmt
    return out


_FMT_ID = {"nvfp4": FMT_NVFP4, "mxfp8": FMT_MXFP8}
_SCRATCH: dict = {}


def _scratch(device, inter: int, hidden: int):
    key = (device, inter, hidden)
    if key not in _SCRATCH:
        _SCRATCH[key] = (
            torch.empty(2 * inter, hidden, device=device, dtype=torch.bfloat16),
            torch.empty(hidden, inter, device=device, dtype=torch.bfloat16),
        )
    return _SCRATCH[key]


class Inkling2XPUExperts(nn.Module):
    """Expert-sharded experts; BF16, or NVFP4 / MXFP8 kept packed (`quant`)."""

    def __init__(self, config, quant: str | None = None) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.quant = quant
        hidden, inter = config.hidden_size, config.intermediate_size
        e = self.num_experts
        if quant is None:
            self.w13_weight = nn.Parameter(
                torch.empty(e, 2 * inter, hidden, dtype=torch.bfloat16), requires_grad=False
            )
            self.w2_weight = nn.Parameter(
                torch.empty(e, hidden, inter, dtype=torch.bfloat16), requires_grad=False
            )
            params = (self.w13_weight, self.w2_weight)
        else:
            pack, group = (2, 16) if quant == "nvfp4" else (1, 32)
            u8 = dict(dtype=torch.uint8)
            self.w13_weight = nn.Parameter(torch.empty(e, 2 * inter, hidden // pack, **u8), requires_grad=False)
            self.w13_weight_scale = nn.Parameter(torch.empty(e, 2 * inter, hidden // group, **u8), requires_grad=False)
            self.w2_weight = nn.Parameter(torch.empty(e, hidden, inter // pack, **u8), requires_grad=False)
            self.w2_weight_scale = nn.Parameter(torch.empty(e, hidden, inter // group, **u8), requires_grad=False)
            params = [self.w13_weight, self.w13_weight_scale, self.w2_weight, self.w2_weight_scale]
            if quant == "nvfp4":
                for nm in ("w13", "w2"):
                    setattr(self, f"{nm}_weight_amax", nn.Parameter(torch.empty(e, dtype=torch.float32), requires_grad=False))
                    setattr(self, f"{nm}_weight_outer_scale_denominator",
                            nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False))
                    params += [getattr(self, f"{nm}_weight_amax"),
                               getattr(self, f"{nm}_weight_outer_scale_denominator")]
            for p in params:
                p.byte_param = True  # raw 1-byte storage; checkpoint dtype (fp8/u8) is reinterpreted
        for param in params:
            set_weight_attrs(param, {"weight_loader": self._load_experts})
        self._packed = False

    def _load_experts(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.shape[0] == self.num_experts * self.ep_size and param.shape[0] == self.num_experts:
            start = self.ep_rank * self.num_experts
            loaded = loaded[start : start + self.num_experts]
        if loaded.dtype != param.dtype:
            loaded = loaded.view(param.dtype)  # 1-byte raw reinterpret (fp8 -> uint8)
        param.data.copy_(loaded)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        if self._packed:
            return
        for name in ("w13_weight", "w13_weight_scale"):
            p = getattr(self, name, None)
            if p is None:
                continue
            e, two_i = p.shape[:2]
            w = p.view(e, two_i // 2, 2, *p.shape[2:])
            p.data = w.transpose(1, 2).reshape(p.shape).contiguous()
        if self.quant == "nvfp4":
            self.w13_outer = (self.w13_weight_amax.data / self.w13_weight_outer_scale_denominator.data).float().contiguous()
            self.w2_outer = (self.w2_weight_amax.data / self.w2_weight_outer_scale_denominator.data).float().contiguous()
        elif self.quant == "mxfp8":
            ones = torch.ones(self.num_experts, device=self.w13_weight.device)
            self.w13_outer = ones
            self.w2_outer = ones.clone()
        self._packed = True

    def expert_bf16(self, e: int, inter: int, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
        """BF16 `(w13 [2I, D], w2 [D, I])` of local expert `e` (dequantised on the fly if packed)."""
        if self.quant is None:
            return self.w13_weight[e], self.w2_weight[e]
        fmt = _FMT_ID[self.quant]
        t13, t2 = _scratch(self.w13_weight.device, inter, hidden)
        dequant_expert(self.w13_weight[e], self.w13_weight_scale[e], float(self.w13_outer[e]), fmt, hidden, t13)
        dequant_expert(self.w2_weight[e], self.w2_weight_scale[e], float(self.w2_outer[e]), fmt, inter, t2)
        return t13, t2


class Inkling2XPUMoE(nn.Module):
    def __init__(self, config, quant: str | None = None) -> None:
        super().__init__()
        self.gate = Inkling2XPUGate(config)
        self.experts = Inkling2XPUExperts(config, quant)
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.inter = config.intermediate_size
        self.hidden = config.hidden_size
        self.gate_softcap = config.mlp_gate_softcap
        self.up_softcap = config.mlp_up_softcap
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def _activation(self, gu: torch.Tensor) -> torch.Tensor:
        if _SITU_OP is not None:
            out = torch.empty_like(gu[..., : gu.shape[-1] // 2])
            _SITU_OP(out, gu, self.gate_softcap, self.up_softcap)
            return out
        return situ_and_mul(gu, self.gate_softcap, self.up_softcap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` `[T, D]` bf16 -> `[T, D]` bf16 (TP all-reduced)."""
        if x.shape[0] <= _SMALL_T and x.device.type == "xpu" and not _TORCH_ROUTER:
            # decode / verify: one fused kernel (gate GEMV + sigmoid + bias + top-k + renorm), int32 idx, fp32 weights
            indices, weights = router_topk(x, self.gate.weight, self.gate.bias, self.route_scale, self.top_k)
            flat = x  # only the shape / device are used below on this path
        else:
            flat = x.float()
            scores = torch.sigmoid(F.linear(flat, self.gate.weight.float()))
            sel = scores + self.gate.bias
            indices = torch.topk(sel, self.top_k, dim=-1, sorted=False).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * self.route_scale

        if _DUMP_DIR and _DUMP_ON[0] and flat.shape[0] >= 1024 and _DUMP_N[0] < 48:
            # debug capture: only while the caller has switched _DUMP_ON on (i.e. around the real
            # generation), never during vLLM's profile/warmup passes (identical dummy tokens).
            torch.save(indices.detach().cpu(), f"{_DUMP_DIR}/real_visit{_DUMP_N[0]:02d}.pt")
            _DUMP_N[0] += 1
        experts = self.experts
        start = experts.ep_rank * experts.num_experts
        if flat.shape[0] <= _SMALL_T and flat.device.type == "xpu" and not _TORCH_MOE:
            if experts.quant is None:
                out = triton_moe_small(
                    x, indices, weights, experts.w13_weight, experts.w2_weight, start,
                    self.gate_softcap, self.up_softcap,
                )
            else:
                out = triton_moe_small_q(
                    x, indices, weights, experts.w13_weight, experts.w13_weight_scale, experts.w13_outer,
                    experts.w2_weight, experts.w2_weight_scale, experts.w2_outer, start,
                    self.gate_softcap, self.up_softcap, _FMT_ID[experts.quant],
                )
            if self.tp_size > 1:
                out = tensor_model_parallel_all_reduce(out)
            return out
        if experts.quant is not None and flat.device.type == "xpu" and not _TORCH_MOE:
            out = triton_moe_grouped_q(
                x, indices, weights, experts.w13_weight, experts.w13_weight_scale, experts.w13_outer,
                experts.w2_weight, experts.w2_weight_scale, experts.w2_outer, start,
                self.gate_softcap, self.up_softcap, _FMT_ID[experts.quant],
            )
            if self.tp_size > 1:
                out = tensor_model_parallel_all_reduce(out)
            return out
        out = torch.zeros_like(flat)
        local = (indices >= start) & (indices < start + experts.num_experts)
        if local.any():
            local_idx = indices - start
            for slot in range(self.top_k):
                sel_e = local_idx[:, slot]
                mask = local[:, slot]
                if not mask.any():
                    continue
                tok = mask.nonzero(as_tuple=True)[0]
                for e in sel_e[tok].unique().tolist():
                    rows = tok[sel_e[tok] == e]
                    w13_e, w2_e = experts.expert_bf16(e, self.inter, self.hidden)
                    gu = flat[rows] @ w13_e.float().t()
                    act = self._activation(gu)
                    y = act.to(torch.bfloat16) @ w2_e.t()
                    out.index_add_(0, rows, y.float() * weights[rows, slot, None])
        out = out.to(torch.bfloat16)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
