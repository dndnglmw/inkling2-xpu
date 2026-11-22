# SPDX-License-Identifier: Apache-2.0
import json, sys, time
import torch
from safetensors import safe_open
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_moe_q import triton_moe_small_q, FMT_NVFP4, FMT_MXFP8
from vllm.models.inkling2.xpu.triton_moe_grouped import triton_moe_grouped_q

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
    for T, K, ep_start, nloc in ((5, 4, 0, NE), (40, 4, 0, 4), (300, 4, 4, 4), (1000, 8, 0, NE)):
        torch.manual_seed(T)
        x = torch.randn(T, D, device=dev).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(2 * NE, device=dev)[:K] for _ in range(T)])
        wts = torch.rand(T, K, device=dev) * 2
        sl = slice(ep_start, ep_start + nloc)
        args = (w13[:nloc], s13[:nloc], o13[:nloc], w2[:nloc], s2[:nloc], o2[:nloc])
        # offset the expert tensors so local expert e uses tensor row e (ep_start applies to ids only)
        ref = triton_moe_small_q(x, idx, wts, *args, ep_start, 4.0, 25.0, fmt)
        out = triton_moe_grouped_q(x, idx, wts, *args, ep_start, 4.0, 25.0, fmt)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-6)).item()
        g = rel < 5e-3; ok &= g
        print(f"{'PASS' if g else 'FAIL'} fmt {fmt} T={T} K={K} ep_start={ep_start} nloc={nloc}: rel {rel:.5f}")
    # timing at a prefill-sized batch
    T, K = 2048, 8
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(NE, device=dev)[:K] for _ in range(T)])
    wts = torch.rand(T, K, device=dev)
    args = (w13, s13, o13, w2, s2, o2)
    for _ in range(2): triton_moe_grouped_q(x, idx, wts, *args, 0, 4.0, 25.0, fmt)
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(3): triton_moe_grouped_q(x, idx, wts, *args, 0, 4.0, 25.0, fmt)
    torch.xpu.synchronize()
    print(f"  fmt {fmt}: T=2048 (all 8 experts local, all pairs valid = 16384 pairs): {(time.time()-t0)/3*1e3:.1f} ms")
print("ALL PASS" if ok else "SOME FAILED")
