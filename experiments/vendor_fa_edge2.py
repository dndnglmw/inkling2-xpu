import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev = torch.device("xpu"); H, Hkv, D, bs = 12, 4, 128, 64
nb = 300
kv = torch.randn(nb, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
table = (torch.arange(3 * 90, dtype=torch.int32).view(3, 90) % nb).to(dev)
def go(qlens, far, max_q):
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=dev)
    q = torch.randn(sum(qlens), H, D, device=dev).to(torch.bfloat16)
    out, lse = flash_attn_varlen_func(q, kv.transpose(1, 2)[..., :D], kv.transpose(1, 2)[..., D:], max_seqlen_q=max_q, cu_seqlens_q=cu,
        max_seqlen_k=90 * 64, seqused_k=torch.tensor(far, dtype=torch.int32, device=dev), softmax_scale=D ** -0.5, causal=False,
        block_table=table[: len(qlens)], return_softmax_lse=True)
    off = 0
    for r, n in enumerate(qlens):
        l = lse[:, off:off + n]; o = out[off:off + n]
        print(f"   req{r} q={n} far={far[r]:5d}: lse min/max {l.min().item():9.2f}/{l.max().item():9.2f}  out finite={bool(torch.isfinite(o).all())}")
        off += n
for qlens, far, mq in (([1, 1, 33], [3952, 0, 3077], 33), ([1, 33], [0, 3077], 33), ([1, 1], [3952, 0], 1), ([1, 1], [0, 3952], 1)):
    print("qlens", qlens, "far", far); go(qlens, far, mq)
