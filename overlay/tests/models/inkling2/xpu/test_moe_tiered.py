# SPDX-License-Identifier: Apache-2.0
"""Tiered grouped MoE vs the verified per-pair kernel, on REAL checkpoint tensors and REAL routing dumps.

Needs ~/work/agents/routing/real_visit*.pt (captured from a natural-text prompt; see agents/routing/capture.py).
"""
import glob
import json
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_moe_grouped import triton_moe_grouped_q
from vllm.models.inkling2.xpu.triton_moe_q import FMT_MXFP8, FMT_NVFP4, triton_moe_small_q

dev = torch.device("xpu")
D, I, EL = 2304, 1152, 64
root = "/work/models/inkling-hf"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]


def get(name, sl):
    with safe_open(f"{root}/{wm[name]}", "pt") as f:
        return f.get_slice(name)[sl]


def perm_rows(t):
    e, r = t.shape[:2]
    return t.view(e, r // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape)


def sync():
    torch.xpu.synchronize()


files = sorted(glob.glob("/work/agents/routing/real_visit*.pt"))
visits = [0, 3, 17, 29, 40]  # incl. the two most skewed visits (17: one expert takes every token)
ok = True
for layer, fmt in ((0, FMT_NVFP4), (21, FMT_MXFP8)):
    pre = f"model.llm.layers.{layer}.mlp.experts."
    w13 = perm_rows(get(pre + "w13_weight", slice(0, EL)).view(torch.uint8)).contiguous().to(dev)
    s13 = perm_rows(get(pre + "w13_weight_scale", slice(0, EL)).view(torch.uint8)).contiguous().to(dev)
    w2 = get(pre + "w2_weight", slice(0, EL)).view(torch.uint8).contiguous().to(dev)
    s2 = get(pre + "w2_weight_scale", slice(0, EL)).view(torch.uint8).contiguous().to(dev)
    if fmt == FMT_NVFP4:
        o13 = (get(pre + "w13_weight_amax", slice(0, EL)) / get(pre + "w13_weight_outer_scale_denominator", slice(0, 1))).float().to(dev)
        o2 = (get(pre + "w2_weight_amax", slice(0, EL)) / get(pre + "w2_weight_outer_scale_denominator", slice(0, 1))).float().to(dev)
    else:
        o13 = torch.ones(EL, device=dev)
        o2 = torch.ones(EL, device=dev)
    args = (w13, s13, o13, w2, s2, o2)
    torch.manual_seed(layer)
    x = torch.randn(2048, D, device=dev).to(torch.bfloat16)
    wts = torch.rand(2048, 8, device=dev) * 2
    for vi in visits:
        idx = torch.load(files[vi]).to(dev)
        for rank in (0, 1):
            ep = rank * EL
            ref = triton_moe_small_q(x, idx, wts, *args, ep, 4.0, 25.0, fmt)
            out = triton_moe_grouped_q(x, idx, wts, *args, ep, 4.0, 25.0, fmt)  # tiered (default)
            rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-6)).item()
            good = rel < 3e-3
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} fmt {fmt} visit {vi:2d} rank {rank}: rel {rel:.5f}")
print("ALL PASS" if ok else "SOME FAILED")
