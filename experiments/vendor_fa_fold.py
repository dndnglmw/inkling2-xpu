"""Fix candidates for the far-region vendor call at q_len>1: (a) one q_len=1 request per verify row, (b) fold rows into the GQA group dim."""
import sys
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = torch.device("xpu")
H, Hkv, D, bs = 12, 4, 128, 64
G = H // Hkv
torch.manual_seed(0)
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 210000
nbl = (ctx + bs - 1) // bs
kv8 = (torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(torch.float8_e4m3fn)
table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
one = torch.ones(1, device=dev, dtype=torch.float32)
kt = kv8.transpose(1, 2)
K, V = kt[..., :D], kt[..., D:]
kw = dict(softmax_scale=D ** -0.5, causal=False, return_softmax_lse=True, k_descale=one, v_descale=one)


def base(q, T):  # current path
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    return flash_attn_varlen_func(q, K, V, max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=ctx,
                                  seqused_k=torch.tensor([ctx], dtype=torch.int32, device=dev), block_table=table, **kw)


def expand(q, T):  # (a)
    cu = torch.arange(T + 1, dtype=torch.int32, device=dev)
    return flash_attn_varlen_func(q, K, V, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=ctx,
                                  seqused_k=torch.full((T,), ctx, dtype=torch.int32, device=dev),
                                  block_table=table.expand(T, -1).contiguous(), **kw)


def fold(q, T):  # (b): [T,H,D] -> [1, Hkv*T*G, D] ordered (kvh, t, g)
    qf = q.view(T, Hkv, G, D).permute(1, 0, 2, 3).reshape(1, Hkv * T * G, D)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    o, l = flash_attn_varlen_func(qf, K, V, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=ctx,
                                  seqused_k=torch.tensor([ctx], dtype=torch.int32, device=dev), block_table=table, **kw)
    o = o.view(Hkv, T, G, D).permute(1, 0, 2, 3).reshape(T, H, D)
    l = l.view(Hkv, T, G).permute(0, 2, 1).reshape(H, T) if l.dim() == 2 else l
    return o, l


def timeit(f):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        f()
    e1.record(); torch.xpu.synchronize()
    return e0.elapsed_time(e1) / 10 * 1000


for T in (2, 4, 8):
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    ob, lb = base(q, T)
    row = f"T={T}: base {timeit(lambda: base(q, T)):7.0f} us |"
    for name, fn in (("expand", expand), ("fold", fold)):
        try:
            o, l = fn(q, T)
            eo = ((o.float() - ob.float()).norm() / ob.float().norm()).item()
            el = (l.float().reshape(lb.shape) - lb.float()).abs().max().item() if l.numel() == lb.numel() else float("nan")
            row += f" {name} {timeit(lambda: fn(q, T)):7.0f} us (out relerr {eo:.1e}, lse maxerr {el:.1e}) |"
        except Exception as e:
            row += f" {name} FAILED {type(e).__name__}: {str(e)[:100]} |"
    print(row, flush=True)
print("lse shape base:", lb.shape)
