# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 cross-pass messages for Intel XPU (pure torch).

Validated against llama.cpp (3-pass top-128 correlation 0.9995).

Per core layer, each of the four sublayer streams (attn, k, v, mlp) receives a
message that is a block-attention readout of the *previous pass's* bank:

    bank = [block[prelude+1], ..., block[last], pass_output]        (2 entries)
    message_s = softmax_q(rms(bank) @ q_s) @ rms(bank)               (fp32)

and the destination is blended in fp32 with a gated low-rank controller:

    pair = flatten([rms(dest); rms(msg)], -1)                        (bf16)
    h    = pair @ W_down.T                                          (bf16)
    out  = (1 - tanh(h @ W_forget.T)) * dest
         + tanh(h @ W_gain.T) * msg                                 (fp32)

The caller then applies the sublayer input norm.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Order of the four message streams and their controllers.
LOOP_STREAMS = ("attn", "k", "v", "mlp")


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)


def attn_res_mix(
    blocks: Sequence[torch.Tensor],
    cur: torch.Tensor,
    query: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """AttnRes mix over `[*blocks, cur]` with a pre-scaled `query` `[Q, D]`.

    Sources are RMS-normalized (unweighted); returns fp32 `[Q, T, D]`.
    """
    states = torch.stack([*blocks, cur], dim=0)  # [S, T, D]
    states = rms_norm(states, eps)
    scores = torch.einsum("q d, s t d -> s q t", query.float(), states)
    probs = torch.softmax(scores, dim=0)
    return torch.einsum("s q t, s t d -> q t d", probs, states)


class LoopStreamController(nn.Module):
    """`(1 - tanh(f)) * dest + tanh(g) * message` for one stream."""

    def __init__(self, hidden_size: int, gate_rank: int) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(gate_rank, 2 * hidden_size))
        self.gain_weight = nn.Parameter(torch.empty(hidden_size, gate_rank))
        self.forget_weight = nn.Parameter(torch.empty(hidden_size, gate_rank))

    def forward(self, dest: torch.Tensor, message: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pair = torch.stack((dest, message), dim=1)  # [T, 2, D]
        pair = rms_norm(pair, eps).to(torch.bfloat16).flatten(1)
        hidden = pair @ self.down_weight.t()
        g = torch.mm(hidden.float(), self.gain_weight.t().float())
        f = torch.mm(hidden.float(), self.forget_weight.t().float())
        return (1.0 - torch.tanh(f)) * dest + torch.tanh(g) * message


class LoopControllers(nn.Module):
    """The four controllers of one loop core layer (checkpoint names)."""

    def __init__(self, hidden_size: int, gate_rank: int) -> None:
        super().__init__()
        for stream in LOOP_STREAMS:
            self.add_module(stream, LoopStreamController(hidden_size, gate_rank))

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        """Stacked copies (stream order attn, k, v, mlp) for the fused XPU kernels."""
        mods = [getattr(self, s) for s in LOOP_STREAMS]
        self.down_all = torch.stack([m.down_weight.data for m in mods]).contiguous()
        self.gain_all = torch.stack([m.gain_weight.data for m in mods]).contiguous()
        self.forget_all = torch.stack([m.forget_weight.data for m in mods]).contiguous()

    def blend(
        self,
        stream: str,
        dest: torch.Tensor,
        message: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return getattr(self, stream)(dest, message, eps)
