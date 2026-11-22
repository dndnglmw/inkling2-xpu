import sys, time
from types import SimpleNamespace
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention
dev = torch.device("xpu")
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024

def bench(ctx, n_q=1, iters=20):
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
    f = lambda: triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, n_q], pos, extent, None, D**-0.5)
    f(); torch.xpu.synchronize()
    t = time.time()
    for _ in range(iters): f()
    torch.xpu.synchronize()
    return (time.time() - t) / iters * 1e3

for ctx in (1024, 8192, 32768, 131072, 262144, 491520):
    print(f"decode ctx={ctx}: {bench(ctx):.3f} ms/layer-call")
for n in (512,):
    print(f"prefill n_q={n} ctx=0: {bench(0, n_q=n, iters=5):.3f} ms/layer-call")
