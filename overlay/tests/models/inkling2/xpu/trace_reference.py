# SPDX-License-Identifier: Apache-2.0
"""Bisect: trace the pure-torch reference (1 pass = non-loop) against hf_trace.json."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm"))
from vllm.models.inkling2.xpu import reference as ref  # noqa: E402

IDS = [371, 151451, 149879, 151194, 149884]

cfg, weights = ref.load_checkpoint(os.path.expanduser("~/work/models/inkling-bf16"))
cfg.num_passes = 1

model = ref.Inkling2Reference(cfg, weights)

trace: list[tuple[float, float]] = []
orig_layer = ref.Inkling2Reference._layer


def traced_layer(self, slot, stream, blocks, positions, messages):
    out = orig_layer(self, slot, stream, blocks, positions, messages)
    t = out[0]
    trace.append((float(t.abs().sum()), float(t.square().sum())))
    return out


ref.Inkling2Reference._layer = traced_layer

ids = torch.tensor(IDS)
logits = model.prefill(ids)

hf = json.load(open("/tmp/hf_trace.json"))
hf_layers = hf["layers"]
print(f"{'i':>2} {'rel_abs':>9} {'rel_sq':>9}")
for i, (a, b) in enumerate(zip(trace, hf_layers[1:])):
    ra = abs(a[0] - b[0]) / max(b[0], 1e-9)
    rq = abs(a[1] - b[1]) / max(b[1], 1e-9)
    flag = "  <-- diverged" if rq > 0.02 else ""
    print(f"{i:>2} {ra:9.4f} {rq:9.4f}{flag}")

e_abs, e_sq = hf_layers[0]
emb = ref.weighted_rms_norm(
    ref.rms_norm(model.w["model.llm.embed.weight"][ids].float(), cfg.rms_eps),
    model.w["model.llm.embed_norm.weight"],
    cfg.rms_eps,
)
t = emb.float()
print(f"embed rel_sq = {abs(float(t.square().sum()) - e_sq) / e_sq:.4f}")

top = torch.topk(logits[-1], 16)
print("ref top16:", list(zip(top.indices.tolist(), [round(float(v), 2) for v in top.values])))
print("hf  top16:", list(zip(hf["logits_top_idx"], [round(v, 2) for v in hf["logits_top"]])))
