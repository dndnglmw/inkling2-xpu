"""Feasibility: vendor flash attention with FP8 (e4m3) K/V in OUR cache layout, bf16 queries, unit descale."""
import time
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = torch.device("xpu")
H, Hkv, D, bs = 12, 4, 128, 64
torch.manual_seed(0)


def build(far, T):
    nbl = (far + bs - 1) // bs
    kv = (torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(torch.bfloat16)
    table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    return kv, table, q


def call(kvc, table, q, far, descale=None):
    T = q.shape[0]
    kt = kvc.transpose(1, 2)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    sk = torch.tensor([far], dtype=torch.int32, device=dev)
    kw = {}
    if descale is not None:
        kw = dict(k_descale=descale, v_descale=descale)
    return flash_attn_varlen_func(q, kt[..., :D], kt[..., D:], max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=far,
                                  seqused_k=sk, softmax_scale=D ** -0.5, causal=False, block_table=table,
                                  return_softmax_lse=True, **kw)


kv, table, q = build(5000, 64)
ref_o, ref_l = call(kv, table, q, 5000)                       # bf16 cache
kv8 = kv.to(torch.float8_e4m3fn)
deq = kv8.to(torch.bfloat16)
exp_o, exp_l = call(deq, table, q, 5000)                      # bf16 cache holding the fp8-rounded values (the exact expectation)
one = torch.ones(1, device=dev, dtype=torch.float32)
for label, ds in (("no descale", None), ("descale=1 (scalar tensor)", one), ("descale=1 [seqs,kvh]", torch.ones(1, Hkv, device=dev, dtype=torch.float32))):
    try:
        o, l = call(kv8, table, q, 5000, ds)
        e1 = ((o.float() - exp_o.float()).norm() / exp_o.float().norm()).item()
        e2 = ((o.float() - ref_o.float()).norm() / ref_o.float().norm()).item()
        print(f"fp8 KV, {label:28s}: out vs fp8-rounded-bf16 rel {e1:.4f} | vs original bf16 rel {e2:.4f} | lse maxdiff {(l - exp_l).abs().max().item():.4f}")
    except Exception as ex:
        print(f"fp8 KV, {label:28s}: FAILED {type(ex).__name__}: {str(ex)[:140]}")


def bench(f, n=8):
    f(); torch.xpu.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.time() - t) / n
for far, T, tag in ((131072, 1, "decode"), (491520, 1, "decode"), (262144, 2048, "prefill")):
    kv, table, q = build(far, T)
    kv8 = kv.to(torch.float8_e4m3fn)
    tb = bench(lambda: call(kv, table, q, far))
    try:
        t8 = bench(lambda: call(kv8, table, q, far, one))
        print(f"{tag:8s} far={far:7d} T={T:4d}: bf16 {tb*1e3:8.3f} ms | fp8 {t8*1e3:8.3f} ms | speedup {tb/t8:4.2f}x")
    except Exception as ex:
        print(f"{tag} far={far}: fp8 FAILED {type(ex).__name__}: {str(ex)[:120]}")
