"""Prefill attention tile-config sweep (INKLING_ATTN_CFG = BLOCK_M,BLOCK_N,warps): local-window and global layer at T=2048 chunk, fp8 KV."""
import os, sys, time
import torch

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention

dev = torch.device("xpu")
H, Hkv, D, bs, extent, T = 12, 4, 128, 64, 1024, 2048
MAXBLK = 262144 // bs
torch.manual_seed(0)


def case(ctx, window):
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
    return lambda: triton_paged_attention(q, rel, sink, gate, kv, table, sl, qsl, [0, T], pos, extent, window, D ** -0.5)


def timeit(f, n=4):
    for _ in range(2):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


loc, glob = case(16000, 1024), case(16000, None)
for cfg in sys.argv[1:]:
    os.environ["INKLING_ATTN_CFG"] = cfg
    try:
        print(f"cfg {cfg:10s}: local {timeit(loc):6.2f} ms | global(near part+vendor far) {timeit(glob):6.2f} ms", flush=True)
    except Exception as e:
        print(f"cfg {cfg:10s}: FAILED {type(e).__name__} {str(e)[:80]}", flush=True)
