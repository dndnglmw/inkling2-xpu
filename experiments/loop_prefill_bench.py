"""pre_attn / pre_mlp ops at prefill chunk size (T=2048): mix_op and ctrl_blend_norm_op per call, TP2 model shapes (D=2304, rank 64)."""
import sys, time
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_loop import mix_op, ctrl_blend_norm_op

dev = torch.device("xpu")
T, D, R = int(sys.argv[1]) if len(sys.argv) > 1 else 2048, 2304, 64
torch.manual_seed(0)


def timeit(f, n=10):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


blocks = lambda n: [torch.randn(T, D, device=dev) for _ in range(n)]
cur = torch.randn(T, D, device=dev)
for name, nb, Q, norm in (("attn mix q3 S5 +norm", 4, 3, True), ("attn mix q1 S5 +norm", 4, 1, True), ("msg mix q4 S3", 2, 4, False), ("mlp mix q1 S5 +norm", 4, 1, True)):
    b = blocks(nb)
    q = torch.randn(Q, D, device=dev) * 0.02
    nw = torch.randn(Q, D, device=dev) if norm else None
    print(f"{name:22s}: {timeit(lambda: mix_op(b, cur, q, norm_weight=nw)):7.3f} ms", flush=True)
for J in (3, 1):
    dest = torch.randn(J, T, D, device=dev)
    msg = torch.randn(J, T, D, device=dev)
    wd = (torch.randn(J, R, 2 * D, device=dev) * 0.02).to(torch.bfloat16)
    wg = (torch.randn(J, D, R, device=dev) * 0.02).to(torch.bfloat16)
    wf = (torch.randn(J, D, R, device=dev) * 0.02).to(torch.bfloat16)
    nw = torch.randn(1, D, device=dev)
    print(f"ctrl_blend_norm J={J}      : {timeit(lambda: ctrl_blend_norm_op(dest, msg, wd, wg, wf, nw, 1e-6, 1e-6)):7.3f} ms", flush=True)
