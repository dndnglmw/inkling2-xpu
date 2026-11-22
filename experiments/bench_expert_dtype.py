"""Decode-time MoE kernel: bf16 experts (no dequant) vs NVFP4 vs MXFP8, real shapes, random top-8 routing.

TP2 = 64 local experts of 128 (about half the pairs are local); TP1 = all 128 local. Times are ms per layer-visit call;
x48 gives ms/token of MoE at that batch size (decode visits = 48 per forward).
"""
import statistics as st
import sys
import time

import torch

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_moe import triton_moe_small
from vllm.models.inkling2.xpu.triton_moe_q import FMT_MXFP8, FMT_NVFP4, triton_moe_small_q

dev = torch.device("xpu")
D, I, K = 2304, 1152, 8
torch.manual_seed(0)


def timed(fn, n=1):
    fn(); torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def routings(T, count=16):
    g = torch.Generator().manual_seed(T)
    return [torch.stack([torch.randperm(128, generator=g)[:K] for _ in range(T)]).to(dev) for _ in range(count)]


def make_weights(EL):
    w = {}
    w["bf16"] = ((torch.randn(EL, 2 * I, D, device=dev) * 0.02).to(torch.bfloat16),
                 (torch.randn(EL, D, I, device=dev) * 0.02).to(torch.bfloat16))
    w["nvfp4"] = (torch.randint(0, 256, (EL, 2 * I, D // 2), dtype=torch.uint8, device=dev),
                  torch.randint(0x30, 0x48, (EL, 2 * I, D // 16), dtype=torch.uint8, device=dev),
                  torch.randint(0, 256, (EL, D, I // 2), dtype=torch.uint8, device=dev),
                  torch.randint(0x30, 0x48, (EL, D, I // 16), dtype=torch.uint8, device=dev))
    w["mxfp8"] = (torch.randint(0, 0x7F, (EL, 2 * I, D), dtype=torch.uint8, device=dev),
                  torch.randint(120, 128, (EL, 2 * I, D // 32), dtype=torch.uint8, device=dev),
                  torch.randint(0, 0x7F, (EL, D, I), dtype=torch.uint8, device=dev),
                  torch.randint(120, 128, (EL, D, I // 32), dtype=torch.uint8, device=dev))
    return w


rows = []
for tp, EL in (("TP2", 64), ("TP1", 128)):
    w = make_weights(EL)
    o = torch.full((EL,), 1e-2, device=dev)
    bytes_per_expert = {"bf16": 3 * D * I * 2, "nvfp4": 3 * D * I // 2 * (1 + 1 / 8), "mxfp8": 3 * D * I * (1 + 1 / 32)}
    for T in (1, 2, 4, 8):
        x = torch.randn(T, D, device=dev).to(torch.bfloat16)
        wts = torch.rand(T, K, device=dev)
        rs = routings(T)
        res = {}
        for name in ("bf16", "nvfp4", "mxfp8"):
            def run(idx, name=name):
                if name == "bf16":
                    return triton_moe_small(x, idx, wts, w["bf16"][0], w["bf16"][1], 0, 4.0, 25.0)
                a, b, c, d = w[name]
                return triton_moe_small_q(x, idx, wts, a, b, o, c, d, o, 0, 4.0, 25.0, FMT_NVFP4 if name == "nvfp4" else FMT_MXFP8)
            ms = [timed(lambda: run(idx), n=3) for idx in rs]
            # local pairs per routing -> bytes actually streamed (per-pair kernels re-read an expert per pair)
            lp = st.mean(int(((idx >= 0) & (idx < EL)).sum()) for idx in rs)
            res[name] = (st.mean(ms), lp, st.mean(ms) * 1e-3 > 0 and lp * bytes_per_expert[name] / (st.mean(ms) * 1e-3) / 1e9)
        rows.append((tp, T, res))
        b, n, m = res["bf16"], res["nvfp4"], res["mxfp8"]
        print(f"{tp} T={T}: local pairs {b[1]:5.1f} | bf16 {b[0]:6.3f} ms ({b[2]:4.0f} GB/s)  nvfp4 {n[0]:6.3f} ms ({n[2]:4.0f} GB/s)  "
              f"mxfp8 {m[0]:6.3f} ms ({m[2]:4.0f} GB/s) | bf16/nvfp4 = {b[0] / n[0]:4.2f}x  -> x48: bf16 {b[0] * 48:5.1f} nvfp4 {n[0] * 48:5.1f} mxfp8 {m[0] * 48:5.1f} ms/token", flush=True)
    del w
    torch.xpu.empty_cache()
