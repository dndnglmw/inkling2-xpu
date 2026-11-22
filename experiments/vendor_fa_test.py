import time, torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev = torch.device("xpu")
H, Hkv, D, bs = 12, 4, 128, 64
def build(ctx_far, T):
    nbl = (ctx_far + bs - 1) // bs
    kv = torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)       # OUR layout [nb, Hkv, bs, 2D]
    table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    return kv, table, q
def vendor(kv, table, q, far_len, kv_scale=None):
    T = q.shape[0]
    k = kv.transpose(1, 2)[..., :D]; v = kv.transpose(1, 2)[..., D:]               # [nb, bs, Hkv, D] strided views of our cache
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    sk = torch.tensor([far_len], dtype=torch.int32, device=dev)
    return flash_attn_varlen_func(q, k, v, max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=far_len, seqused_k=sk,
                                  softmax_scale=D ** -0.5, causal=False, block_table=table, return_softmax_lse=True)
def ref(kv, table, q, far_len):
    nbl = (far_len + bs - 1) // bs
    blocks = table[0, :nbl].long()
    kk = kv[blocks][..., :D].transpose(1, 2).reshape(-1, Hkv, D)[:far_len]          # [S, Hkv, D]
    vv = kv[blocks][..., D:].transpose(1, 2).reshape(-1, Hkv, D)[:far_len]
    kk = kk.float().repeat_interleave(H // Hkv, 1); vv = vv.float().repeat_interleave(H // Hkv, 1)
    s = torch.einsum("t h d, s h d -> h t s", q.float(), kk) * D ** -0.5
    lse = torch.logsumexp(s, -1)                                                    # [H, T]
    return torch.einsum("h t s, s h d -> t h d", torch.softmax(s, -1), vv), lse
# correctness (small)
kv, table, q = build(3000, 64)
try:
    out, lse = vendor(kv, table, q, 3000)
    ro, rl = ref(kv, table, q, 3000)
    print("out", tuple(out.shape), "lse", tuple(lse.shape), "| out rel err %.4f" % ((out.float() - ro).norm() / ro.norm()).item(),
          "| lse max abs err %.4f" % (lse.float().reshape(H, -1)[:, :64] - rl).abs().max().item() if lse.numel() == H * 64 else "lse shape " + str(tuple(lse.shape)))
except Exception as e:
    print("vendor call FAILED:", type(e).__name__, str(e)[:300])
    raise SystemExit
# speed
def bench(f, n=5):
    f(); torch.xpu.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.time() - t) / n
for far, T in ((16384, 2048), (65536, 2048), (262144, 2048)):
    kv, table, q = build(far, T)
    dt = bench(lambda: vendor(kv, table, q, far))
    print(f"prefill far={far:7d} T={T}: {dt*1e3:8.2f} ms  {T*far*H*D*4/dt/1e12:6.1f} TFLOP/s")
for far in (16384, 131072, 262144, 524288):
    kv, table, q = build(far, 1)
    dt = bench(lambda: vendor(kv, table, q, far), 10)
    bytes_ = far * Hkv * 2 * D * 2
    print(f"decode  far={far:7d} T=1   : {dt*1e3:8.3f} ms  KV read {bytes_/dt/1e9:6.0f} GB/s")
