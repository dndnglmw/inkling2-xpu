"""Decode-size MoE (NVFP4, TP2: 64 local experts) per-visit time at T=1..8 over INKLING_MOEQ_SPLIT configs. Real-style random top-8 routing."""
import os, sys, time
import torch

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_moe_q import FMT_NVFP4, triton_moe_small_q

dev = torch.device("xpu")
D, I, K, EL = 2304, 1152, 8, 64
torch.manual_seed(0)
w13 = torch.randint(0, 256, (EL, 2 * I, D // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(0x30, 0x48, (EL, 2 * I, D // 16), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (EL, D, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(0x30, 0x48, (EL, D, I // 16), dtype=torch.uint8, device=dev)
o = torch.full((EL,), 1e-2, device=dev)


def routings(T, count=24):
    g = torch.Generator().manual_seed(T)
    return [torch.stack([torch.randperm(128, generator=g)[:K] for _ in range(T)]).to(dev) for _ in range(count)]


def run(T, n=3):
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    wts = torch.rand(T, K, device=dev)
    rs = routings(T)
    f = lambda r: triton_moe_small_q(x, r, wts, w13, s13, o, w2, s2, o, 0, 4.0, 25.0, FMT_NVFP4)
    for r in rs[:2]:
        f(r)
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        for r in rs:
            f(r)
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / (n * len(rs)) * 1e3


cfgs = sys.argv[1:] or ["default"]
for c in cfgs:
    if c == "default":
        os.environ.pop("INKLING_MOEQ_SPLIT", None)
    else:
        os.environ["INKLING_MOEQ_SPLIT"] = c
    try:
        print(f"{c:28s} " + " | ".join(f"T={T} {run(T):.3f}" for T in (2, 4, 8)) + " ms/visit", flush=True)
    except Exception as e:
        print(f"{c:28s} FAILED {type(e).__name__} {str(e)[:70]}", flush=True)
