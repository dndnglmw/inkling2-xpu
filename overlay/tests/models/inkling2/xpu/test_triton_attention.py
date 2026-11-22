# SPDX-License-Identifier: Apache-2.0
"""Triton XPU attention kernel vs the torch `_paged_attention` (run on an XPU)."""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.attention as A
from vllm.models.inkling2.xpu.attention import _paged_attention, _dense_attention
A._USE_TRITON = False  # reference = pure torch
from vllm.models.inkling2.xpu.triton_attention import triton_paged_attention

dev = torch.device("xpu")


def run(case, H, Hkv, D, bs, extent, window, seqs, fp8=False):
    """seqs: list of (context_len, new_tokens). fp8: cache is float8_e4m3fn; K/V/cache are pre-rounded to
    fp8-representable values so the bf16 torch reference sees exactly the data the kernel reads."""
    torch.manual_seed(0)
    R = len(seqs)
    qsl = [0]
    for c, n in seqs:
        qsl.append(qsl[-1] + n)
    T = qsl[-1]
    seq_lens = [c + n for c, n in seqs]
    max_blocks = max((s + bs - 1) // bs for s in seq_lens) + 1
    nb = R * max_blocks + 2
    table = torch.arange(R * max_blocks, dtype=torch.int32).view(R, max_blocks) + 1
    kv0 = torch.randn(nb, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
    q = torch.randn(T, H, D, device=dev).to(torch.bfloat16)
    k = torch.randn(T, Hkv, D, device=dev).to(torch.bfloat16)
    v = torch.randn(T, Hkv, D, device=dev).to(torch.bfloat16)
    if fp8:
        r8 = lambda t: t.to(torch.float8_e4m3fn).to(torch.bfloat16)
        kv0, k, v = r8(kv0), r8(k), r8(v)
    rel_states = torch.randn(T, H, 16, device=dev).to(torch.bfloat16)
    rel_proj = torch.randn(16, extent, device=dev) * 0.3
    gate = torch.randn(T, H * D, device=dev).to(torch.bfloat16)
    sink = torch.randn(T, H, device=dev).to(torch.bfloat16)
    pos, slots = [], []
    for r, (c, n) in enumerate(seqs):
        for i in range(n):
            p = c + i
            pos.append(p)
            slots.append(int(table[r, p // bs]) * bs + p % bs)
    pos_t = torch.tensor(pos, device=dev)
    md = SimpleNamespace(
        block_table=table.to(dev),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev),
    )
    scaling = D**-0.5
    kv_a = kv0.clone()
    ref = _paged_attention(q, k, v, rel_states, gate, sink, rel_proj, kv_a, md, pos_t, extent, window, scaling)
    if len(seqs) == 1 and seqs[0][0] == 0:  # exact cache-free reference
        ref = _dense_attention(q, k, v, rel_states, gate, sink, rel_proj, pos_t, extent, window, scaling)
    kv_b = kv0.clone()
    sl = md.slot_mapping
    kv_b[sl // bs, :, sl % bs, :] = torch.cat([k.float(), v.float()], -1).to(kv_b.dtype)
    rel = torch.einsum("t h d, d e -> t h e", rel_states.float(), rel_proj.float()).contiguous()
    if fp8:
        kv_b = kv_b.to(torch.float8_e4m3fn)  # exact: every value is already fp8-representable
    out = triton_paged_attention(q, rel, sink, gate, kv_b, md.block_table, md.seq_lens,
                                 md.query_start_loc, qsl, pos_t, extent, window, scaling)
    err = (out.float() - ref.float()).abs().max().item()
    rel_err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    ok = rel_err < 2e-2
    print(f"{'PASS' if ok else 'FAIL'} {case}: max_abs {err:.4f} rel {rel_err:.5f}")
    return ok


ok = True
ok &= run("decode full", 8, 2, 128, 16, 1024, None, [(100, 1), (37, 1)])
ok &= run("prefill full", 8, 2, 128, 16, 1024, None, [(0, 70), (0, 33)])
# bs == BLOCK_N exercises the descriptor (block-load) far path used in production
ok &= run("prefill bs=64 desc", 12, 4, 128, 64, 1024, None, [(0, 256), (2000, 70)])
ok &= run("decode bs=64 desc", 12, 4, 128, 64, 1024, None, [(5000, 1), (200, 1)])
ok &= run("chunked prefill ctx", 8, 2, 128, 16, 1024, None, [(50, 40)])
ok &= run("prefill+decode mix", 8, 2, 128, 16, 1024, None, [(0, 20), (90, 1)])
ok &= run("swa decode", 8, 2, 128, 16, 64, 64, [(200, 1)])
ok &= run("swa prefill", 8, 2, 128, 16, 64, 64, [(0, 150)])
ok &= run("tiny D=16 prefill", 4, 2, 16, 16, 32, 32, [(0, 50)])
ok &= run("tiny D=16 decode", 4, 2, 16, 16, 32, 32, [(20, 1), (45, 1)])
ok &= run("extent<seq full", 8, 2, 128, 16, 32, None, [(0, 100)])
ok &= run("FAR long decode (vendor far + near)", 12, 4, 128, 64, 1024, None, [(5000, 1), (3300, 1)])
ok &= run("FAR long prefill chunk", 12, 4, 128, 64, 1024, None, [(4200, 70)])
ok &= run("FAR mixed batch (short + long + chunk)", 12, 4, 128, 64, 1024, None, [(5000, 1), (40, 1), (4100, 33)])
ok &= run("FAR boundary (ctx = extent+few)", 12, 4, 128, 64, 1024, None, [(1030, 1), (1100, 4)])
ok &= run("FAR spec-verify 8 tokens", 12, 4, 128, 64, 1024, None, [(6000, 8)])
# ---- FP8 (e4m3, unit-scale) KV cache ----
ok &= run("FP8 decode", 8, 2, 128, 16, 1024, None, [(100, 1), (37, 1)], fp8=True)
ok &= run("FP8 prefill full", 8, 2, 128, 16, 1024, None, [(0, 70), (0, 33)], fp8=True)
ok &= run("FP8 chunked prefill ctx", 8, 2, 128, 16, 1024, None, [(50, 40)], fp8=True)
ok &= run("FP8 swa decode", 8, 2, 128, 16, 64, 64, [(200, 1)], fp8=True)
ok &= run("FP8 swa prefill", 8, 2, 128, 16, 64, 64, [(0, 150)], fp8=True)
ok &= run("FP8 bs=64 near-only path", 12, 4, 128, 64, 1024, None, [(2000, 70)], fp8=True)
ok &= run("FP8 FAR long decode (vendor fp8 far + near)", 12, 4, 128, 64, 1024, None, [(5000, 1), (3300, 1)], fp8=True)
ok &= run("FP8 FAR long prefill chunk", 12, 4, 128, 64, 1024, None, [(4200, 70)], fp8=True)
ok &= run("FP8 FAR mixed batch", 12, 4, 128, 64, 1024, None, [(5000, 1), (40, 1), (4100, 33)], fp8=True)
ok &= run("FP8 FAR spec-verify 8 tokens", 12, 4, 128, 64, 1024, None, [(6000, 8)], fp8=True)
# far-region fold / expand (q_len 2..16 -> q_len=1 vendor sweeps)
ok &= run("FAR verify uniform 4 rows x3 reqs (fold)", 12, 4, 128, 64, 1024, None, [(6000, 4), (3300, 4), (1100, 4)])
ok &= run("FAR verify ragged (expand)", 12, 4, 128, 64, 1024, None, [(6000, 4), (3300, 2), (1100, 3)])
ok &= run("FP8 FAR verify uniform 4 rows x3 reqs (fold)", 12, 4, 128, 64, 1024, None, [(6000, 4), (3300, 4), (1100, 4)], fp8=True)
ok &= run("FP8 FAR verify ragged (expand)", 12, 4, 128, 64, 1024, None, [(6000, 4), (3300, 2), (1100, 3)], fp8=True)
ok &= run("FAR verify 16 rows (expand, T*G>48)", 12, 4, 128, 64, 1024, None, [(6000, 16), (4000, 16)])

# ---- kv_write into an fp8 cache: matches a plain cast, clamps out-of-range values, skips slot -1 ----
from vllm.models.inkling2.xpu.triton_prep import kv_write
kk = (torch.randn(6, 4, 128, device=dev) * 3).to(torch.bfloat16)
vv = torch.randn(6, 4, 128, device=dev).to(torch.bfloat16)
kk[0, 0, 0] = 1000.0  # out of e4m3 range -> must clamp to 448, not NaN
cache = torch.zeros(5, 4, 16, 256, device=dev, dtype=torch.float8_e4m3fn)
slots = torch.tensor([3, 17, -1, 40, 79, 20], device=dev)
kv_write(kk, vv, cache, slots)
ref = torch.zeros(5, 4, 16, 256, device=dev, dtype=torch.float8_e4m3fn)
for t in range(6):
    s = int(slots[t])
    if s >= 0:
        ref[s // 16, :, s % 16, :128] = kk[t].float().clamp(-448, 448).to(torch.float8_e4m3fn)
        ref[s // 16, :, s % 16, 128:] = vv[t].float().clamp(-448, 448).to(torch.float8_e4m3fn)
wr_ok = torch.equal(cache.view(torch.uint8), ref.view(torch.uint8))
print("PASS fp8 kv_write (cast + clamp + skip)" if wr_ok else "FAIL fp8 kv_write")
ok &= wr_ok
print("ALL PASS" if ok else "SOME FAILED")
