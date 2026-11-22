"""Depth-level numerical gate for the far-region fold: fold vs old q_len>1 path vs fp32 reference, FP8 cache, TP2 head shapes.
Realistic-ish stats: K unit-clipped-like (randn*0.8), V RMS-normalised. Reports output rel error, LSE error and the KL of an attention-weights
proxy is not needed: the far output feeds a softmax-merge, so out/LSE error is the right quantity."""
import sys
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

dev = torch.device("xpu")
H, Hkv, D, bs = 12, 4, 128, 64
G = H // Hkv
one = torch.ones(1, device=dev, dtype=torch.float32)
kw = dict(softmax_scale=D ** -0.5, causal=False, return_softmax_lse=True, k_descale=one, v_descale=one)
torch.manual_seed(0)


def ref(q, K32, V32):  # fp32 reference over the cache as stored (dequantised fp8)
    qf = q.float().view(q.shape[0], Hkv, G, D)
    s = torch.einsum("thgd,nhd->thgn", qf, K32) * D ** -0.5
    lse = torch.logsumexp(s, dim=-1)
    o = torch.einsum("thgn,nhd->thgd", torch.softmax(s, dim=-1), V32)
    return o.reshape(q.shape[0], H, D), lse.reshape(q.shape[0], H)


worst = 0
for ctx in (64000, 134000, 258000):
    nbl = (ctx + bs - 1) // bs
    kv8 = (torch.randn(nbl + 2, Hkv, bs, 2 * D, device=dev) * 0.8).to(torch.float8_e4m3fn)
    table = (torch.arange(nbl, dtype=torch.int32) + 1).view(1, -1).to(dev)
    kt = kv8.transpose(1, 2)
    K, V = kt[..., :D], kt[..., D:]
    kv32 = kv8[1 : nbl + 1].float().permute(0, 2, 1, 3).reshape(-1, Hkv, 2 * D)[:ctx]  # [ctx, Hkv, 2D]
    K32, V32 = kv32[..., :D], kv32[..., D:]
    for T in (2, 4):
        q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
        sk = torch.tensor([ctx], dtype=torch.int32, device=dev)
        o_old, l_old = flash_attn_varlen_func(q, K, V, max_seqlen_q=T, cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32, device=dev),
                                              max_seqlen_k=ctx, seqused_k=sk, block_table=table, **kw)
        qf = q.view(T, Hkv, G, D).permute(1, 0, 2, 3).reshape(1, Hkv * T * G, D)
        o_f, l_f = flash_attn_varlen_func(qf, K, V, max_seqlen_q=1, cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=dev),
                                          max_seqlen_k=ctx, seqused_k=sk, block_table=table, **kw)
        o_f = o_f.view(Hkv, T, G, D).permute(1, 0, 2, 3).reshape(T, H, D)
        l_f = l_f.view(Hkv, T, G).permute(1, 0, 2).reshape(T, H)  # [T, H]
        o_r, l_r = ref(q, K32, V32)
        l_old = l_old.t()  # [H,T] -> [T,H]
        rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
        e_old, e_fold, e_cross = rel(o_old, o_r), rel(o_f, o_r), rel(o_f, o_old)
        le_old, le_fold = (l_old.float() - l_r).abs().max().item(), (l_f.float() - l_r).abs().max().item()
        worst = max(worst, e_fold)
        print(f"ctx {ctx:6d} T={T}: out relerr vs fp32 ref  old {e_old:.2e}  fold {e_fold:.2e} | fold-vs-old {e_cross:.2e} | LSE maxerr vs ref old {le_old:.1e} fold {le_fold:.1e}", flush=True)
print(f"worst fold out relerr vs fp32 reference: {worst:.2e} (bf16 output rounding is ~2e-3)")
