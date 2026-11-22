# SPDX-License-Identifier: Apache-2.0
"""Triton XPU short-conv kernel vs the torch reference `short_conv`."""
import sys
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.sconv import short_conv
from vllm.models.inkling2.xpu.triton_sconv import triton_short_conv

dev = torch.device("xpu")


def run(case, heads, width, dynamic, seqs, addend, residual, pad=0, K=4, bs=4):
    torch.manual_seed(1)
    dim = heads * width
    R = len(seqs)
    qsl = [0]
    for c, n in seqs:
        qsl.append(qsl[-1] + n)
    T = qsl[-1]
    maxb = max((c + n + bs - 1) // bs for c, n in seqs) + 1
    nb = R * maxb + 2
    table = (torch.arange(R * maxb, dtype=torch.int32).view(R, maxb) + 1).to(dev)
    C = 2 * 128 + 2 * 64  # cache channel width, stream at an offset
    offset = 64
    cache0 = torch.randn(nb, heads, bs, max(C, offset + width), device=dev).to(torch.bfloat16)
    x = torch.randn(T + pad, dim, device=dev).to(torch.bfloat16)
    pos, slots, seq_idx, qstart = [], [], [], []
    for r, (c, n) in enumerate(seqs):
        for i in range(n):
            p = c + i
            pos.append(p)
            slots.append(int(table[r, p // bs]) * bs + p % bs)
            seq_idx.append(r)
            qstart.append(qsl[r])
    for _ in range(pad):
        pos.append(0); slots.append(-1); seq_idx.append(-1); qstart.append(-1)
    pos_t = torch.tensor(pos, device=dev)
    slots_t = torch.tensor(slots, dtype=torch.int64, device=dev)
    seq_t = torch.tensor(seq_idx, dtype=torch.int32, device=dev)
    qs_t = torch.tensor(qstart, dtype=torch.int32, device=dev)
    if dynamic:
        weight = torch.randn(K, dim, device=dev).to(torch.bfloat16)
        dyn = torch.randn(T + pad, heads * K, device=dev).to(torch.bfloat16)
    else:
        weight = torch.randn(dim, K, device=dev).to(torch.bfloat16)
        dyn = None
    add = torch.randn(T + pad, dim, device=dev).to(torch.bfloat16) if addend else None
    res = torch.randn(T + pad, dim, device=dev).to(torch.bfloat16) if residual else None
    kw = dict(dynamic_taps=dyn, heads=heads, kernel=K, addend=add, residual=res)
    ca, cb = cache0.clone(), cache0.clone()
    ref = short_conv(x, weight, ca, pos_t, table, slots_t, seq_t, qs_t, offset, width, **kw)
    out = triton_short_conv(x, weight, cb, pos_t, table, slots_t, seq_t, qs_t, offset, width, **kw)
    real = slice(0, T)
    err = (out[real].float() - ref[real].float()).abs().max().item()
    cache_ok = torch.equal(ca, cb)
    ok = err < 0.05 and cache_ok
    print(f"{'PASS' if ok else 'FAIL'} {case}: max_abs {err:.4f} cache_equal {cache_ok}")
    return ok


ok = True
ok &= run("static prefill", 2, 128, False, [(0, 9), (0, 5)], False, False)
ok &= run("static decode w/ ctx", 2, 128, False, [(10, 1), (3, 1)], False, False)
ok &= run("static chunk ctx", 2, 128, False, [(6, 7)], False, False)
ok &= run("dynamic prefill", 4, 128, True, [(0, 9), (0, 5)], False, False)
ok &= run("dynamic decode", 4, 128, True, [(10, 1), (3, 1)], False, False)
ok &= run("hidden stream addend+resid", 1, 512, False, [(0, 6), (7, 1)], True, True)
ok &= run("addend only + padding", 1, 512, False, [(5, 1), (2, 1)], True, False, pad=3)
# ---- bit-exactness of the fusions: strided input, fp32 output, fused cache write ----
def exact(case, heads, width, dynamic, seqs, addend, residual, K=4, bs=4):
    torch.manual_seed(2)
    dim = heads * width
    R = len(seqs)
    qsl = [0]
    for c, n in seqs:
        qsl.append(qsl[-1] + n)
    T = qsl[-1]
    maxb = max((c + n + bs - 1) // bs for c, n in seqs) + 1
    table = (torch.arange(R * maxb, dtype=torch.int32).view(R, maxb) + 1).to(dev)
    cache0 = torch.randn(R * maxb + 2, heads, bs, max(2 * 128 + 2 * 64, 64 + width), device=dev).to(torch.bfloat16)
    wide = torch.randn(T, dim + 96, device=dev).to(torch.bfloat16)  # x is a row-strided slice, like a split of qkvr
    x_strided = wide[:, 16:16 + dim]
    x_contig = x_strided.contiguous()
    pos, slots, seq_idx, qstart = [], [], [], []
    for r, (c, n) in enumerate(seqs):
        for i in range(n):
            p = c + i
            pos.append(p); slots.append(int(table[r, p // bs]) * bs + p % bs); seq_idx.append(r); qstart.append(qsl[r])
    pos_t = torch.tensor(pos, device=dev)
    slots_t = torch.tensor(slots, dtype=torch.int64, device=dev)
    seq_t = torch.tensor(seq_idx, dtype=torch.int32, device=dev)
    qs_t = torch.tensor(qstart, dtype=torch.int32, device=dev)
    if dynamic:
        weight = torch.randn(K, dim, device=dev).to(torch.bfloat16)
        dyn = torch.randn(T, heads * K, device=dev).to(torch.bfloat16)
    else:
        weight = torch.randn(dim, K, device=dev).to(torch.bfloat16)
        dyn = None
    add = torch.randn(T, dim, device=dev).to(torch.bfloat16) if addend else None
    res = torch.randn(T, dim, device=dev).float() if residual else None
    kw = dict(dynamic_taps=dyn, heads=heads, kernel=K, addend=add, residual=res)
    c1, c2, c3 = cache0.clone(), cache0.clone(), cache0.clone()
    o_ref = triton_short_conv(x_contig, weight, c1, pos_t, table, slots_t, seq_t, qs_t, 64, width, **kw)
    o_str = triton_short_conv(x_strided, weight, c2, pos_t, table, slots_t, seq_t, qs_t, 64, width, **kw)
    o_f32 = triton_short_conv(x_strided, weight, c3, pos_t, table, slots_t, seq_t, qs_t, 64, width, out_dtype=torch.float32, **kw)
    good = torch.equal(o_ref, o_str) and torch.equal(o_ref.float(), o_f32) and torch.equal(c1, c2) and torch.equal(c1, c3) and o_f32.dtype == torch.float32
    print(f"{'PASS' if good else 'FAIL'} exact {case}")
    return good


ok &= exact("decode static hidden (addend+resid)", 1, 512, False, [(10, 4), (3, 1)], True, True)
ok &= exact("decode dynamic kv", 4, 128, True, [(10, 4)], False, False)
ok &= exact("tile prefill static (addend+resid)", 1, 512, False, [(300, 200)], True, True)
ok &= exact("tile multi-request", 2, 128, False, [(0, 90), (50, 70)], False, False)

# tile (prefill) path: T >= 128, static taps
ok &= run("tile: single chunk with ctx", 1, 512, False, [(300, 200)], False, False)
ok &= run("tile: multi-request chunks", 2, 128, False, [(0, 90), (50, 70), (7, 3)], False, False)
ok &= run("tile: hidden addend+resid", 1, 512, False, [(1000, 160), (2, 1)], True, True)
ok &= run("tile: addend only + padding", 1, 512, False, [(40, 150)], True, False, pad=5)
ok &= run("tile: odd T not a multiple of 16", 1, 384, False, [(0, 133)], True, True, bs=16)
print("ALL PASS" if ok else "SOME FAILED")
