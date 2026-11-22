import sys, torch
sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.attention as A
from vllm.models.inkling2.xpu.attention import _paged_attention
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention
from types import SimpleNamespace
dev = torch.device("xpu"); torch.manual_seed(0)
H, Hkv, D, bs, extent = 12, 4, 128, 64, 1024
def run(seqs):
    R = len(seqs); qsl = [0]
    for c, n in seqs: qsl.append(qsl[-1] + n)
    T = qsl[-1]; seq_lens = [c + n for c, n in seqs]
    maxb = max((s + bs - 1) // bs for s in seq_lens) + 1
    table = (torch.arange(R * maxb, dtype=torch.int32).view(R, maxb) + 1)
    kv0 = torch.randn(R * maxb + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16); k = torch.randn(T, Hkv, D, device=dev).to(torch.bfloat16); v = torch.randn(T, Hkv, D, device=dev).to(torch.bfloat16)
    rs = torch.randn(T, H, 16, device=dev).to(torch.bfloat16); rp = torch.randn(16, extent, device=dev) * 0.3
    gate = torch.randn(T, H * D, device=dev).to(torch.bfloat16); sink = torch.randn(T, H, device=dev).to(torch.bfloat16)
    pos, slots = [], []
    for r, (c, n) in enumerate(seqs):
        for i in range(n):
            p = c + i; pos.append(p); slots.append(int(table[r, p // bs]) * bs + p % bs)
    pos_t = torch.tensor(pos, device=dev)
    md = SimpleNamespace(block_table=table.to(dev), seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev), query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev))
    A._USE_TRITON = False
    ref = _paged_attention(q, k, v, rs, gate, sink, rp, kv0.clone(), md, pos_t, extent, None, D ** -0.5)
    kvb = kv0.clone(); sl = md.slot_mapping
    kvb[sl // bs, :, sl % bs, :] = torch.cat([k.float(), v.float()], -1).to(kvb.dtype)
    rel = torch.einsum("t h d, d e -> t h e", rs.float(), rp.float()).contiguous()
    out = triton_paged_attention(q, rel, sink, gate, kvb, md.block_table, md.seq_lens, md.query_start_loc, qsl, pos_t, extent, None, D ** -0.5)
    for r in range(R):
        a, b = qsl[r], qsl[r + 1]
        e = ((out[a:b].float() - ref[a:b].float()).norm() / ref[a:b].float().norm()).item()
        print(f"  req{r} ctx={seqs[r]} rel {e:.4f}")
for seqs in ([(5000, 1), (40, 1), (4100, 33)], [(5000, 1), (4100, 33)], [(40, 1), (4100, 33)], [(4100, 33), (5000, 1)]):
    print("batch", seqs); run(seqs)
