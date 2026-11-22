# SPDX-License-Identifier: Apache-2.0
"""Fused router kernel vs the torch router it replaces."""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_router import router_topk

dev = torch.device("xpu")
D, E, K, SCALE = 2304, 128, 8, 8.0
torch.manual_seed(0)
gate_w = (torch.randn(E, D, device=dev) * 0.05).to(torch.bfloat16)
bias = torch.randn(E, device=dev) * 0.1


def ref(x):
    scores = torch.sigmoid(F.linear(x.float(), gate_w.float()))
    sel = scores + bias
    idx = torch.topk(sel, K, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE
    return idx, w


ok = True
for T in (1, 2, 5, 8, 16):
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    ri, rw = ref(x)
    ki, kw = router_topk(x, gate_w, bias, SCALE, K)
    same_set = torch.equal(torch.sort(ri.int(), 1).values, torch.sort(ki, 1).values)
    # compare weights per expert id (order differs)
    rmap = torch.zeros(T, E, device=dev).scatter_(1, ri, rw)
    kmap = torch.zeros(T, E, device=dev).scatter_(1, ki.long(), kw)
    werr = (rmap - kmap).abs().max().item()
    good = same_set and werr < 1e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} T={T}: expert sets equal {same_set}, max |w diff| {werr:.2e}")


def bench(f, n=300):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e6


x = torch.randn(1, D, device=dev).to(torch.bfloat16)
print(f"T=1: torch router {bench(lambda: ref(x)):6.1f} us | fused {bench(lambda: router_topk(x, gate_w, bias, SCALE, K)):6.1f} us (eager, incl. launch)")
print("ALL PASS" if ok else "SOME FAILED")
