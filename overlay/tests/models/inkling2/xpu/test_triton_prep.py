# SPDX-License-Identifier: Apache-2.0
import sys
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.attention import _l2_clip, _v_norm
from vllm.models.inkling2.xpu.triton_prep import rows_op, kv_write

dev = torch.device("xpu")
torch.manual_seed(0)
ok = True
def chk(name, a, b):
    global ok
    d = (a.float() - b.float()).abs().max().item()
    g = d <= 0.0160  # <= 1 bf16 ulp at magnitude ~2
    ok &= g
    print(f"{'PASS' if g else 'FAIL'} {name}: max_abs {d:.5f} exact={torch.equal(a, b)}")

T = 5
proj = torch.randn(T, 1800, device=dev).to(torch.bfloat16)   # strided views like the fused projection
q = proj[:, :12 * 128].view(T, 12, 128)
r = proj[:, 12 * 128:12 * 128 + 12 * 16].view(T, 12, 16)
qs = (torch.rand(12, 128, device=dev) + 0.5)
rs = (torch.rand(12, 16, device=dev) + 0.5)
tau = 1 + 0.1 * torch.rand(T, device=dev)
chk("q clip+scale", rows_op(q, mode=0, threshold=22.627, scale=qs), _l2_clip(q, 22.627, qs))
chk("q clip+scale+tau", rows_op(q, mode=0, threshold=22.627, scale=qs, tau=tau),
    (_l2_clip(q, 22.627, qs).float() * tau[:, None, None]).to(torch.bfloat16))
chk("r clip+scale+tau", rows_op(r, mode=0, threshold=8.0, scale=rs, tau=tau),
    (_l2_clip(r, 8.0, rs).float() * tau[:, None, None]).to(torch.bfloat16))
k = torch.randn(T, 4, 128, device=dev).to(torch.bfloat16) * 30
chk("k clip", rows_op(k, mode=0, threshold=22.627), _l2_clip(k, 22.627))
chk("v rms", rows_op(k, mode=1, eps=1e-6), _v_norm(k, 1e-6))

nb, bs = 6, 16
cache = torch.zeros(nb, 4, bs, 256, device=dev, dtype=torch.bfloat16)
ref = cache.clone()
v = torch.randn(T, 4, 128, device=dev).to(torch.bfloat16)
slots = torch.tensor([3, 17, -1, 40, 95], device=dev)
kv_write(k, v, cache, slots)
for t in range(T):
    s = int(slots[t])
    if s >= 0:
        ref[s // bs, :, s % bs, :128] = k[t]; ref[s // bs, :, s % bs, 128:] = v[t]
print("PASS kv_write" if torch.equal(cache, ref) else "FAIL kv_write"); ok &= torch.equal(cache, ref)
print("ALL PASS" if ok else "SOME FAILED")
