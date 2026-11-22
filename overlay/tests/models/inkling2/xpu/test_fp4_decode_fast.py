"""The bit-trick e2m1 / e4m3 decoders are bit-exact twins of the reference decoders over every input."""
import sys

import torch

sys.path.insert(0, "/work/vllm")
from vllm.triton_utils import tl, triton
from vllm.models.inkling2.xpu.triton_moe_q import _e2m1, _e2m1_fast, _e4m3, _e4m3_fast

dev = torch.device("xpu")


@triton.jit
def _k(In, Ref, New, N, KIND: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(In + i, mask=i < N, other=0).to(tl.int32)
    if KIND == 0:
        a, b = _e2m1(v), _e2m1_fast(v)
    else:
        a, b = _e4m3(v), _e4m3_fast(v)
    tl.store(Ref + i, a, mask=i < N)
    tl.store(New + i, b, mask=i < N)


ok = True
for name, kind, n in (("e2m1", 0, 16), ("e4m3", 1, 256)):
    x = torch.arange(n, dtype=torch.int32, device=dev)
    ref = torch.empty(n, device=dev)
    new = torch.empty(n, device=dev)
    _k[(1,)](x, ref, new, n, KIND=kind, BLOCK=256)
    same = bool(((ref == new) | (torch.isnan(ref) & torch.isnan(new))).all())
    ok &= same
    print(f"{'PASS' if same else 'FAIL'} {name}: all {n} inputs equal")
print("ALL PASS" if ok else "SOME FAILED")
