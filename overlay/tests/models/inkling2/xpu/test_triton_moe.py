# SPDX-License-Identifier: Apache-2.0
"""Triton small-batch MoE vs torch math (packed [gates; ups] w13, SITU)."""
import sys, time
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.moe import situ_and_mul
from vllm.models.inkling2.xpu.triton_moe import triton_moe_small

dev = torch.device("xpu")
D, I, E, EL, K = 2304, 1152, 128, 64, 8
torch.manual_seed(0)
w13 = (torch.randn(EL, 2 * I, D, device=dev) * 0.02).to(torch.bfloat16)
w2 = (torch.randn(EL, D, I, device=dev) * 0.02).to(torch.bfloat16)


def ref(x, idx, wts, start):
    flat = x.float()
    out = torch.zeros_like(flat)
    for t in range(x.shape[0]):
        for k in range(K):
            e = int(idx[t, k]) - start
            if not (0 <= e < EL):
                continue
            gu = flat[t:t + 1] @ w13[e].float().t()
            act = situ_and_mul(gu, 4.0, 25.0)
            y = act.to(torch.bfloat16) @ w2[e].t()
            out[t] += (y.float() * wts[t, k])[0]
    return out.to(torch.bfloat16)


ok = True
for T, start in [(1, 0), (3, 64), (8, 0)]:
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(T)])
    wts = torch.rand(T, K, device=dev) * 2
    r = ref(x, idx, wts, start)
    o = triton_moe_small(x, idx, wts, w13, w2, start, 4.0, 25.0)
    rel = ((o.float() - r.float()).norm() / r.float().norm().clamp(min=1e-6)).item()
    good = rel < 1e-2
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} T={T} ep_start={start}: rel {rel:.5f} (|ref|={r.float().norm():.3f})")
x = torch.randn(1, D, device=dev).to(torch.bfloat16)
idx = torch.randperm(E, device=dev)[:K].view(1, K) % EL
wts = torch.rand(1, K, device=dev)
for _ in range(3): triton_moe_small(x, idx, wts, w13, w2, 0, 4.0, 25.0)
torch.xpu.synchronize(); t = time.time()
for _ in range(20): triton_moe_small(x, idx, wts, w13, w2, 0, 4.0, 25.0)
torch.xpu.synchronize()
print(f"decode T=1, 8 local experts: {(time.time()-t)/20*1e3:.3f} ms")
print("ALL PASS" if ok else "SOME FAILED")
