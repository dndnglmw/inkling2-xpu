"""Hidden-stream short conv (attn_sconv / mlp_sconv shapes) at a T=2048 prefill chunk: us per call, write kernel vs conv kernel."""
import sys, time
import torch
sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.triton_sconv as S

dev = torch.device("xpu")
T, D, K, bs, ctx = 2048, 2304, 4, int(sys.argv[1]) if len(sys.argv) > 1 else 16, 30000
nblk = (ctx + T + bs - 1) // bs + 2
table = (torch.arange(nblk, dtype=torch.int32) + 1).view(1, -1).to(dev)
cache = torch.randn(nblk + 4, 1, bs, 2 * D, device=dev).to(torch.bfloat16)
pos = torch.arange(ctx, ctx + T, device=dev)
slots = (table[0, pos // bs].long() * bs + pos % bs)
seq_idx = torch.zeros(T, dtype=torch.int32, device=dev)
qstart = torch.zeros(T, dtype=torch.int32, device=dev)
x = torch.randn(T, D, device=dev).to(torch.bfloat16)
w = torch.randn(D, K, device=dev).to(torch.bfloat16)
add = torch.randn(T, D, device=dev).to(torch.bfloat16)
res = torch.randn(T, D, device=dev).float()


def timeit(f, n=10):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


kw = dict(dynamic_taps=None, heads=1, kernel=K)
print(f"attn_sconv shape (no addend/res): {timeit(lambda: S.triton_short_conv(x, w, cache, pos, table, slots, seq_idx, qstart, 0, D, addend=None, residual=None, **kw)):.3f} ms")
print(f"mlp_sconv shape (addend+fp32 resid): {timeit(lambda: S.triton_short_conv(x, w, cache, pos, table, slots, seq_idx, qstart, D, D, addend=add, residual=res, **kw)):.3f} ms")
print(f"floor: one read x + one write out (bf16) = {2 * T * D * 2 / 1e6:.0f} MB -> {2 * T * D * 2 / 400e9 * 1e3:.3f} ms at 400 GB/s")
