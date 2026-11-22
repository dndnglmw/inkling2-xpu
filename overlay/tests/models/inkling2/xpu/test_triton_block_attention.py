# SPDX-License-Identifier: Apache-2.0
import sys
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.dspark import block_attention
from vllm.models.inkling2.xpu.triton_block_attention import triton_block_attention
dev = torch.device("xpu")
ok = True
for name, reqs, lens, window in (("global", 2, [300, 77], 0), ("local win", 2, [2500, 90], 1032), ("long global", 1, [30000], 0), ("short", 3, [8, 20, 1100], 1032)):
    torch.manual_seed(0)
    H, Hkv, D, bs, block = 12, 4, 128, 64, 8
    maxb = max((L + bs - 1) // bs for L in lens) + 1
    table = (torch.arange(len(lens) * maxb, dtype=torch.int32).view(len(lens), maxb) + 1).to(dev)
    kv = torch.randn(len(lens) * maxb + 2, Hkv, bs, 2 * D, device=dev).to(torch.bfloat16)
    q = torch.randn(len(lens) * block, H, D, device=dev).to(torch.bfloat16)
    sl = torch.tensor(lens, dtype=torch.int32, device=dev)
    ref = block_attention(q, kv, table, sl, block, window, D ** -0.5)
    out = triton_block_attention(q, kv, table, sl, block, window, D ** -0.5)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    g = rel < 1e-2; ok &= g
    print(f"{'PASS' if g else 'FAIL'} {name}: rel {rel:.5f}")
print("ALL PASS" if ok else "SOME FAILED")
