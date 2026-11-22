"""Per-component context cost at TP2 shapes: target global-layer attention wrapper (fp8 KV, T=4 verify rows) and the draft global block attention
(bf16 draft KV, 8 rows, window 0) at short vs long context. Reports us/call and the delta, x visits/step."""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention
from vllm.models.inkling2.xpu.triton_block_attention import triton_block_attention

dev = torch.device("xpu")
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024
MAXLEN = 262144
MAXBLK = MAXLEN // bs
torch.manual_seed(0)


def timeit(f, n=20):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    torch.xpu.synchronize()
    return e0.elapsed_time(e1) / n * 1000


def target_case(ctx, T=4, dtype=torch.float8_e4m3fn):
    nblk = (ctx + T + bs - 1) // bs
    kv = (torch.randn(nblk + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(dtype)
    table = torch.zeros(1, MAXBLK, dtype=torch.int32, device=dev)  # table width = max-model-len, like the server
    table[0, :nblk] = torch.arange(1, nblk + 1, dtype=torch.int32, device=dev)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    rel = torch.randn(T, H, extent, device=dev) * 0.1
    sink = torch.randn(T, H, device=dev).to(torch.bfloat16)
    gate = torch.randn(T, H * D, device=dev).to(torch.bfloat16)
    pos = torch.arange(ctx, ctx + T, device=dev)
    sl = torch.tensor([ctx + T], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    return lambda: triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, T], pos, extent, None, D ** -0.5)


def draft_case(ctx, window=0, block=8):
    nblk = (ctx + block + bs - 1) // bs
    kv = torch.randn(nblk + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
    table = torch.zeros(1, MAXBLK, dtype=torch.int32, device=dev)
    table[0, :nblk] = torch.arange(1, nblk + 1, dtype=torch.int32, device=dev)
    q = torch.randn(block, H, D, device=dev).to(torch.bfloat16)
    sl = torch.tensor([ctx + block], dtype=torch.int32, device=dev)
    return lambda: triton_block_attention(q, kv, table, sl, block, window, D ** -0.5)


res = {}
for ctx in (2000, 67000, 134000, 220000):
    res[ctx] = (timeit(target_case(ctx)), timeit(draft_case(ctx)), timeit(draft_case(ctx, window=1024 + 8)))
    print(f"ctx {ctx:6d}: target global attn (T=4, fp8) {res[ctx][0]:7.0f} us | draft global block attn (bf16) {res[ctx][1]:7.0f} us | draft local (window) {res[ctx][2]:6.0f} us", flush=True)
b = res[2000]
for ctx in (67000, 134000, 220000):
    dt = res[ctx][0] - b[0]
    dd = res[ctx][1] - b[1]
    print(f"ctx {ctx:6d}: context-dependent step cost = target {dt / 1000 * 8:5.1f} ms (x8 global visits) + draft {dd / 1000 * 1:5.1f} ms (x1 global draft layer) = {(dt * 8 + dd) / 1000:5.1f} ms")
