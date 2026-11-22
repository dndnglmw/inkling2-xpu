"""Cost of one 2048-token prefill chunk by component at TP2 shapes: dense GEMMs, attention (local window / global near+far) at depth."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention

dev = torch.device("xpu")
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024
T, HID = 2048, 2304
MAXBLK = 262144 // bs
torch.manual_seed(0)


def timeit(f, n=5):
    for _ in range(2):
        f()
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    torch.xpu.synchronize()
    return e0.elapsed_time(e1) / n


# dense: qkvr (q 1536 + rel 12*16... approx fused width) and wo, bf16
qkvr_w = torch.randn(5000, HID, device=dev).to(torch.bfloat16)
wo_w = torch.randn(HID, H * D, device=dev).to(torch.bfloat16)
x = torch.randn(T, HID, device=dev).to(torch.bfloat16)
a = torch.randn(T, H * D, device=dev).to(torch.bfloat16)
g_qkvr = timeit(lambda: F.linear(x, qkvr_w))
g_wo = timeit(lambda: F.linear(a, wo_w))
flops = 2 * T * HID * 5000 + 2 * T * H * D * HID
print(f"dense GEMMs per visit: qkvr {g_qkvr:.2f} ms + wo {g_wo:.2f} ms = {g_qkvr + g_wo:.2f} ms ({flops / (g_qkvr + g_wo) / 1e9:.0f} TFLOP/s)")


def attn(ctx, window):
    nblk = (ctx + T + bs - 1) // bs
    kv = (torch.randn(nblk + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(torch.float8_e4m3fn)
    table = torch.zeros(1, MAXBLK, dtype=torch.int32, device=dev)
    table[0, :nblk] = torch.arange(1, nblk + 1, dtype=torch.int32, device=dev)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    rel = torch.randn(T, H, extent, device=dev) * 0.1
    sink = torch.randn(T, H, device=dev).to(torch.bfloat16)
    gate = torch.randn(T, H * D, device=dev).to(torch.bfloat16)
    pos = torch.arange(ctx, ctx + T, device=dev)
    sl = torch.tensor([ctx + T], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    return timeit(lambda: triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, T], pos, extent, window, D ** -0.5), n=3)


for ctx in (16000, 105000, 210000):
    loc, glob = attn(ctx, 1024), attn(ctx, None)
    # 48 visits: local layers vs global (8 global visits: 5, 11x3, 17x3, 23); 40 local
    print(f"attention at ctx {ctx:6d}: local {loc:6.2f} ms | global {glob:7.2f} ms  -> per chunk: 40 local {40 * loc:6.0f} ms + 8 global {8 * glob:6.0f} ms = {40 * loc + 8 * glob:6.0f} ms", flush=True)
