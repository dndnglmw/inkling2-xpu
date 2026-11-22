"""All-reduce cost at prefill chunk size (2048 x 2304 bf16) across the 2 GPUs. Run: torchrun --nproc-per-node 2 ar_bench.py"""
import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.xpu.set_device(rank)
dist.init_process_group("xccl" if hasattr(dist, "is_xccl_available") and dist.is_xccl_available() else "ccl")
dev = torch.device("xpu", rank)
for T in (2048, 4, 1):
    x = torch.randn(T, 2304, device=dev).to(torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(x)
    torch.xpu.synchronize()
    t = time.perf_counter()
    n = 20
    for _ in range(n):
        dist.all_reduce(x)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t) / n * 1e3
    if rank == 0:
        mb = x.numel() * 2 / 1e6
        print(f"T={T:5d} ({mb:5.1f} MB): {ms:6.2f} ms/all-reduce | x96 per chunk = {ms * 96:6.0f} ms", flush=True)
