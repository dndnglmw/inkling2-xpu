"""Large-T controller blend (bmm + fused blend/norm) vs the per-token GEMV kernels, plus timing."""
import sys
import time

import torch

sys.path.insert(0, "/work/vllm")
import vllm.models.inkling2.xpu.triton_loop as L

dev = torch.device("xpu")
D, R = 2304, 64
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


for J, expand, T in ((3, False, 2048), (3, True, 2048), (1, False, 2048), (3, False, 300), (1, False, 70)):
    torch.manual_seed(0)
    dest = torch.randn(1 if expand else J, T, D, device=dev)
    if expand:
        dest = dest.expand(J, T, D)
    msg = torch.randn(J, T, D, device=dev)
    wd = (torch.randn(J, R, 2 * D, device=dev) * 0.02).to(torch.bfloat16)
    wg = (torch.randn(J, D, R, device=dev) * 0.02).to(torch.bfloat16)
    wf = (torch.randn(J, D, R, device=dev) * 0.02).to(torch.bfloat16)
    nw = torch.randn(1, D, device=dev)
    args = (dest, msg, wd, wg, wf, nw, 1e-6, 1e-6)
    L._LARGE_T = 10**9
    ref = L.ctrl_blend_norm_op(*args)
    t_ref = timeit(lambda: L.ctrl_blend_norm_op(*args))
    L._LARGE_T = 1
    new = L.ctrl_blend_norm_op(*args)
    t_new = timeit(lambda: L.ctrl_blend_norm_op(*args))
    rel = ((new.float() - ref.float()).norm() / ref.float().norm()).item()
    good = rel < 1e-2
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} J={J} expand={expand} T={T}: rel {rel:.2e} | gemv {t_ref:6.3f} ms -> gemm {t_new:6.3f} ms ({t_ref / t_new:4.1f}x)")
print("ALL PASS" if ok else "SOME FAILED")
