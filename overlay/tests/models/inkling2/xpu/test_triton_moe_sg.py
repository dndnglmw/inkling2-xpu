# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped small-batch MoE vs the verified per-pair kernel."""
import json, sys, time
import torch
from safetensors import safe_open
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_moe_q import triton_moe_small_q, FMT_NVFP4, FMT_MXFP8
from vllm.models.inkling2.xpu.triton_moe_sg import triton_moe_small_grouped_q

dev = torch.device("xpu")
D, I, NE = 2304, 1152, 8
root = "/work/models/inkling-hf"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
def get(name, sl):
    with safe_open(f"{root}/{wm[name]}", "pt") as f:
        return f.get_slice(name)[sl]
def perm_rows(t):
    e, r = t.shape[:2]
    return t.view(e, r // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape)
ok = True
for layer, fmt in ((0, FMT_NVFP4), (21, FMT_MXFP8)):
    pre = f"model.llm.layers.{layer}.mlp.experts."
    w13 = perm_rows(get(pre + "w13_weight", slice(0, NE)).view(torch.uint8)).contiguous().to(dev)
    s13 = perm_rows(get(pre + "w13_weight_scale", slice(0, NE)).view(torch.uint8)).contiguous().to(dev)
    w2 = get(pre + "w2_weight", slice(0, NE)).view(torch.uint8).contiguous().to(dev)
    s2 = get(pre + "w2_weight_scale", slice(0, NE)).view(torch.uint8).contiguous().to(dev)
    if fmt == FMT_NVFP4:
        o13 = (get(pre + "w13_weight_amax", slice(0, NE)) / get(pre + "w13_weight_outer_scale_denominator", slice(0, 1))).float().to(dev)
        o2 = (get(pre + "w2_weight_amax", slice(0, NE)) / get(pre + "w2_weight_outer_scale_denominator", slice(0, 1))).float().to(dev)
    else:
        o13 = torch.ones(NE, device=dev); o2 = torch.ones(NE, device=dev)
    args = (w13, s13, o13, w2, s2, o2)
    for T, K, ep_start in ((2, 4, 0), (5, 4, 4), (8, 8, 0), (16, 8, 0), (1, 8, 0)):
        torch.manual_seed(T)
        x = torch.randn(T, D, device=dev).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(16, device=dev)[:K] for _ in range(T)])
        wts = torch.rand(T, K, device=dev) * 2
        ref = triton_moe_small_q(x, idx, wts, *args, ep_start, 4.0, 25.0, fmt)
        out = triton_moe_small_grouped_q(x, idx, wts, *args, ep_start, 4.0, 25.0, fmt)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-6)).item()
        g = rel < 2e-3; ok &= g
        print(f"{'PASS' if g else 'FAIL'} fmt {fmt} T={T} K={K} ep_start={ep_start}: rel {rel:.6f}")
# realistic timing: 64 local experts (random valid bytes), T=8 top-8 of 128
EL = 64
w13 = torch.randint(0, 256, (EL, 2 * I, D // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(0x30, 0x48, (EL, 2 * I, D // 16), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (EL, D, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(0x30, 0x48, (EL, D, I // 16), dtype=torch.uint8, device=dev)
o = torch.full((EL,), 1e-2, device=dev)
def bench(f, n=20):
    for _ in range(3): f()
    torch.xpu.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.time() - t) / n * 1e3
for T in (1, 2, 4, 8, 16):
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(128, device=dev)[:8] for _ in range(T)]); wts = torch.rand(T, 8, device=dev)
    a = bench(lambda: triton_moe_small_q(x, idx, wts, w13, s13, o, w2, s2, o, 0, 4.0, 25.0, FMT_NVFP4))
    b = bench(lambda: triton_moe_small_grouped_q(x, idx, wts, w13, s13, o, w2, s2, o, 0, 4.0, 25.0, FMT_NVFP4))
    print(f"T={T:2d} (64 local experts, top-8 of 128): per-pair {a:.3f} ms | expert-grouped {b:.3f} ms")
print("ALL PASS" if ok else "SOME FAILED")
