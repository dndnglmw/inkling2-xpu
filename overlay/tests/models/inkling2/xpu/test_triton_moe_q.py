# SPDX-License-Identifier: Apache-2.0
"""Quantised MoE kernels vs fp32 reference dequant on real Inkling2 tensors."""
import json, sys, time
import torch
from safetensors import safe_open
sys.path.insert(0, "/work/vllm")
sys.path.insert(0, "/work/vllm/scripts")
from inkling2_dequant import dequant_nvfp4, dequant_mxfp8
from vllm.models.inkling2.xpu.moe import situ_and_mul
from vllm.models.inkling2.xpu.triton_moe_q import triton_moe_small_q, dequant_expert, FMT_NVFP4, FMT_MXFP8

dev = torch.device("xpu")
D, I, NE = 2304, 1152, 4
root = "/work/models/inkling-hf"
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]


def get(name, sl):
    with safe_open(f"{root}/{wm[name]}", "pt") as f:
        return f.get_slice(name)[sl]


def perm_rows(t):  # interleaved (g0,u0,g1,u1..) -> [gates; ups] on dim 1
    e, r = t.shape[:2]
    return t.view(e, r // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape)


ok = True
for layer, fmt in ((0, FMT_NVFP4), (21, FMT_MXFP8)):
    pre = f"model.llm.layers.{layer}.mlp.experts."
    w13 = get(pre + "w13_weight", slice(0, NE)); s13 = get(pre + "w13_weight_scale", slice(0, NE))
    w2 = get(pre + "w2_weight", slice(0, NE)); s2 = get(pre + "w2_weight_scale", slice(0, NE))
    if fmt == FMT_NVFP4:
        a13 = get(pre + "w13_weight_amax", slice(0, NE)); d13 = get(pre + "w13_weight_outer_scale_denominator", slice(0, 1))
        a2 = get(pre + "w2_weight_amax", slice(0, NE)); d2 = get(pre + "w2_weight_outer_scale_denominator", slice(0, 1))
        f13 = dequant_nvfp4(w13, s13, a13, d13); f2 = dequant_nvfp4(w2, s2, a2, d2)
        o13 = (a13 / d13).float(); o2 = (a2 / d2).float()
    else:
        f13 = dequant_mxfp8(w13, s13); f2 = dequant_mxfp8(w2, s2)
        o13 = torch.ones(NE); o2 = torch.ones(NE)
    f13 = perm_rows(f13)
    w13p = perm_rows(w13.view(torch.uint8)).contiguous().to(dev)
    s13p = perm_rows(s13.view(torch.uint8)).contiguous().to(dev)
    w2d = w2.view(torch.uint8).contiguous().to(dev); s2d = s2.view(torch.uint8).contiguous().to(dev)
    # --- decode kernel
    torch.manual_seed(0)
    T, K = 2, 2
    x = torch.randn(T, D).to(torch.bfloat16)
    idx = torch.tensor([[0, 3], [2, 1]]); wts = torch.tensor([[1.3, 0.7], [0.4, 1.6]])
    ref = torch.zeros(T, D)
    for t in range(T):
        for k in range(K):
            e = int(idx[t, k])
            gu = x[t].float() @ f13[e].t()
            act = situ_and_mul(gu[None], 4.0, 25.0).to(torch.bfloat16).float()
            ref[t] += (act @ f2[e].t())[0].to(torch.bfloat16).float() * wts[t, k]
    out = triton_moe_small_q(x.to(dev), idx.to(dev), wts.to(dev), w13p, s13p, o13.to(dev), w2d, s2d, o2.to(dev),
                             0, 4.0, 25.0, fmt)
    rel = ((out.cpu().float() - ref.to(torch.bfloat16).float()).norm() / ref.norm()).item()
    g = rel < 1.5e-2; ok &= g
    print(f"{'PASS' if g else 'FAIL'} layer {layer} fmt {fmt} decode kernel: rel {rel:.5f}")
    # --- dequant kernel
    for e in range(NE):
        kdim = D
        outb = torch.empty(2 * I, kdim, device=dev, dtype=torch.bfloat16)
        dequant_expert(w13p[e], s13p[e], float(o13[e]), fmt, kdim, outb)
        r2 = ((outb.cpu().float() - f13[e]).norm() / f13[e].norm()).item()
        outb2 = torch.empty(D, I, device=dev, dtype=torch.bfloat16)
        dequant_expert(w2d[e], s2d[e], float(o2[e]), fmt, I, outb2)
        r3 = ((outb2.cpu().float() - f2[e]).norm() / f2[e].norm()).item()
        g = r2 < 5e-3 and r3 < 5e-3; ok &= g
        print(f"{'PASS' if g else 'FAIL'}   expert {e} dequant: w13 rel {r2:.5f} w2 rel {r3:.5f}")
    # timing
    xd, idd, wd = x.to(dev)[:1], torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=dev), torch.rand(1, 8, device=dev)
    for _ in range(3): triton_moe_small_q(xd, idd, wd, w13p, s13p, o13.to(dev), w2d, s2d, o2.to(dev), 0, 4.0, 25.0, fmt)
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(20): triton_moe_small_q(xd, idd, wd, w13p, s13p, o13.to(dev), w2d, s2d, o2.to(dev), 0, 4.0, 25.0, fmt)
    torch.xpu.synchronize()
    print(f"  fmt {fmt}: decode T=1, 8 experts: {(time.time()-t0)/20*1e3:.3f} ms")
print("ALL PASS" if ok else "SOME FAILED")
