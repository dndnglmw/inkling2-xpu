import sys, time
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention
dev = torch.device("xpu")
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024

def run(ctx, n_q, window=None, iters=3):
    nbl = (ctx + n_q + bs - 1) // bs
    kv = torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
    table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
    q = torch.randn(n_q, H, D, device=dev).to(torch.bfloat16)
    rel = torch.randn(n_q, H, extent, device=dev)
    sink = torch.randn(n_q, H, device=dev).to(torch.bfloat16)
    gate = torch.randn(n_q, H * D, device=dev).to(torch.bfloat16)
    sl = torch.tensor([ctx + n_q], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, n_q], dtype=torch.int32, device=dev)
    pos = torch.arange(ctx, ctx + n_q, device=dev)
    f = lambda: triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, n_q], pos, extent, window, D**-0.5)
    f(); torch.xpu.synchronize(); t = time.time()
    for _ in range(iters): f()
    torch.xpu.synchronize(); dt = (time.time() - t) / iters
    keys = (ctx + n_q / 2) if window is None else min(window, ctx + n_q / 2)
    flops = n_q * keys * H * D * 4
    return dt * 1e3, flops / dt / 1e12

for ctx, n_q, win in ((16384, 2048, None), (65536, 2048, None), (262144, 2048, None), (491520, 2048, None), (16384, 2048, 1024), (0, 2048, None)):
    ms, tf = run(ctx, n_q, win)
    print(f"ctx={ctx:6d} n_q={n_q} window={win}: {ms:8.1f} ms  {tf:5.2f} TFLOP/s")
