import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev = torch.device("xpu"); H, Hkv, D, bs = 12, 4, 128, 64
nb = 80
kv = torch.randn(nb, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
table = torch.stack([torch.arange(1, 41, dtype=torch.int32), torch.arange(41, 81, dtype=torch.int32) % nb]).to(dev)
# 2 requests: first has 0 far keys, second 1500; queries: 3 + 5 tokens
q = torch.randn(8, H, D, device=dev).to(torch.bfloat16)
cu = torch.tensor([0, 3, 8], dtype=torch.int32, device=dev)
for lens in ([0, 1500], [1500, 0], [0, 0], [5, 1500]):
    sk = torch.tensor(lens, dtype=torch.int32, device=dev)
    try:
        out, lse = flash_attn_varlen_func(q, kv.transpose(1, 2)[..., :D], kv.transpose(1, 2)[..., D:], max_seqlen_q=5, cu_seqlens_q=cu,
                                          max_seqlen_k=2560, seqused_k=sk, softmax_scale=D ** -0.5, causal=False, block_table=table,
                                          return_softmax_lse=True)
        a = out[:3]; b = out[3:]
        print(f"far lens {lens}: out finite req0 {bool(torch.isfinite(a).all())} req1 {bool(torch.isfinite(b).all())} | "
              f"lse req0 min/max {lse[:, :3].min().item():.2f}/{lse[:, :3].max().item():.2f} | lse req1 {lse[:, 3:].min().item():.2f}/{lse[:, 3:].max().item():.2f} | |out0|max {a.abs().max().item():.3f}")
    except Exception as e:
        print(f"far lens {lens}: FAILED {type(e).__name__}: {str(e)[:150]}")
