"""Cost of 96 decode-sized (4 x 2304 bf16) all-reduces inside an XPU graph replay, 2 ranks. Run: torchrun --nproc-per-node 2 ar_graph_bench.py"""
import os, time
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.xpu.set_device(rank)
dist.init_process_group("xccl")
dev = torch.device("xpu", rank)
x = torch.randn(4, 2304, device=dev).to(torch.bfloat16)
w = torch.randn(4, 2304, device=dev).to(torch.bfloat16)


def body(n_ar, between):
    y = x.clone()
    for _ in range(n_ar):
        for _ in range(between):
            y = y + w
        dist.all_reduce(y)
    return y


def timeit(f, n=10):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for n_ar, between in ((96, 0), (96, 10), (0, 960)):
    body(n_ar, between); torch.xpu.synchronize()
    try:
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            out = body(n_ar, between)
        torch.xpu.synchronize()
        t_g = timeit(g.replay)
    except Exception as e:
        t_g = float("nan")
        if rank == 0:
            print("graph capture failed:", type(e).__name__, str(e)[:100])
    t_e = timeit(lambda: body(n_ar, between))
    if rank == 0:
        print(f"HOST_WAIT={os.environ.get('VLLM_XPU_ALLREDUCE_HOST_WAIT', '0')} {n_ar:3d} all-reduces + {between * max(n_ar, 1):4d} tiny kernels: graph replay {t_g:7.3f} ms | eager {t_e:7.3f} ms", flush=True)
