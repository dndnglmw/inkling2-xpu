"""attn-res mix kernel at decode (T=1): time per call for the shapes the model uses."""
import os, sys, time
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_loop import mix_op, _MIX_CFG

dev = torch.device("xpu")
D = 2304
cases = {"attn q3 S5 +norm": (4, 3, True), "attn q1 S4 +norm": (3, 1, True), "msg q4 S3": (2, 4, False), "mlp q1 S5 +norm": (4, 1, True)}
total = 0
for name, (nb, Q, norm) in cases.items():
    blocks = [torch.randn(1, D, device=dev) for _ in range(nb)]
    cur = torch.randn(1, D, device=dev)
    q = torch.randn(Q, D, device=dev) * 0.02
    nw = torch.randn(Q, D, device=dev) if norm else None
    f = lambda: mix_op(blocks, cur, q, norm_weight=nw)
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(200): f()
    torch.xpu.synchronize(); us = (time.perf_counter() - t) / 200 * 1e6
    total += us
    print(f"  {name:18s}: {us:6.1f} us")
print(f"cfg {_MIX_CFG}: sum of the 4 shapes {total:6.1f} us")
