"""Vendor flash attention (far region) time vs q_len at depth: does verify (q_len>1) share the KV sweep or not?"""
import sys
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = torch.device("xpu")
H, Hkv, D, bs = 12, 4, 128, 64
torch.manual_seed(0)
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 210000
nbl = (ctx + bs - 1) // bs
kv8 = (torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(torch.float8_e4m3fn)
table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
one = torch.ones(1, device=dev, dtype=torch.float32)
kt = kv8.transpose(1, 2)
bytes_kv = nbl * Hkv * bs * 2 * D  # fp8
print(f"ctx {ctx}: {nbl} blocks, KV read once = {bytes_kv / 1e6:.0f} MB")


def run(T, nreq=1):
    q = torch.randn(T * nreq, H, D, device=dev).to(torch.bfloat16)
    cu = (torch.arange(nreq + 1, dtype=torch.int32) * T).to(dev)
    sk = torch.full((nreq,), ctx, dtype=torch.int32, device=dev)
    tb = table.expand(nreq, -1).contiguous()
    f = lambda: flash_attn_varlen_func(q, kt[..., :D], kt[..., D:], max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=ctx, seqused_k=sk,
                                       softmax_scale=D ** -0.5, causal=False, block_table=tb, return_softmax_lse=True, k_descale=one, v_descale=one)
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        f()
    e1.record(); torch.xpu.synchronize()
    return e0.elapsed_time(e1) / 10


base = None
for T in (1, 2, 3, 4, 5, 8, 16):
    ms = run(T)
    base = base or ms
    print(f"q_len {T:2d}: {ms * 1000:8.0f} us/layer-call | x{ms / base:4.1f} vs q_len 1 | {bytes_kv / ms / 1e6:6.0f} GB/s effective single-sweep")
