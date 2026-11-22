"""Multi-pointer prefill mix kernel (no stacked bank copy) vs the stacked per-token kernel, plus timing."""
import sys
import time

import torch

sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.triton_loop as L

dev = torch.device("xpu")
D = 2304
ok = True


def timeit(f, n=10):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for name, T, nb, Q, norm in (
    ("attn q3 S5 +norm", 2048, 4, 3, True),
    ("attn q1 S5 +norm", 2048, 4, 1, True),
    ("msg q4 S3", 2048, 2, 4, False),
    ("mlp q1 S5 +norm", 2048, 4, 1, True),
    ("attn q1 S2 +norm", 300, 1, 1, True),
    ("attn q3 S6 +norm odd T", 133, 5, 3, True),
    ("first layer-ish S1", 512, 0, 1, True),
):
    torch.manual_seed(0)
    blocks = [torch.randn(T, D, device=dev) for _ in range(nb)]
    cur = torch.randn(T, D, device=dev)
    q = torch.randn(Q, D, device=dev) * 0.02
    nw = torch.randn(Q, D, device=dev) if norm else None
    L._MP_MIN_T = 10**9
    ref = L.mix_op(blocks, cur, q, norm_weight=nw)
    t_ref = timeit(lambda: L.mix_op(blocks, cur, q, norm_weight=nw))
    L._MP_MIN_T = 1
    new = L.mix_op(blocks, cur, q, norm_weight=nw)
    t_new = timeit(lambda: L.mix_op(blocks, cur, q, norm_weight=nw))
    rel = ((new.float() - ref.float()).norm() / ref.float().norm()).item()
    tol = 1e-2 if norm else 1e-5
    good = rel < tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name:24s} T={T}: rel {rel:.1e} | stacked {t_ref:6.3f} ms -> multi-ptr {t_new:6.3f} ms ({t_ref / t_new:4.1f}x)")
print("ALL PASS" if ok else "SOME FAILED")
