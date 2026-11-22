# SPDX-License-Identifier: Apache-2.0
import sys, time
import torch
sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.loop import attn_res_mix, LoopStreamController
from vllm.models.inkling2.xpu.model import _rms_norm
from vllm.models.inkling2.xpu.triton_loop import mix_op, ctrl_blend_norm_op

dev = torch.device("xpu")
torch.manual_seed(0)
D, R = 2304, 64
ok = True
def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()
def chk(name, a, b, tol=2e-3):
    global ok
    e = rel(a, b); g = e < tol; ok &= g
    print(f"{'PASS' if g else 'FAIL'} {name}: rel {e:.6f}")

for T in (1, 5):
    for S, Q in ((2, 1), (4, 3), (5, 4)):
        blocks = [torch.randn(T, D, device=dev) for _ in range(S - 1)]
        cur = torch.randn(T, D, device=dev)
        qry = torch.randn(Q, D, device=dev) * D ** -0.5 * 3
        ref = attn_res_mix(blocks, cur, qry)
        chk(f"mix T={T} S={S} Q={Q}", mix_op(blocks, cur, qry), ref, 1e-5)
        wn = (1 + 0.1 * torch.randn(Q, D, device=dev))
        refn = torch.stack([_rms_norm(ref[j], wn[j], 1e-6) for j in range(Q)])
        chk(f"mix+norm T={T} S={S} Q={Q}", mix_op(blocks, cur, qry, norm_weight=wn), refn, 5e-3)
        w1 = wn[:1]
        refn1 = torch.stack([_rms_norm(ref[j], w1[0], 1e-6) for j in range(Q)])
        chk(f"mix+norm(shared w) T={T} S={S} Q={Q}", mix_op(blocks, cur, qry, norm_weight=w1), refn1, 5e-3)

J = 3
for T in (1, 5):
    ctrls = []
    for _ in range(J):
        c = LoopStreamController(D, R).to(dev).to(torch.bfloat16)
        for p in c.parameters(): p.data.normal_(0, 0.03)
        ctrls.append(c)
    mixed = torch.randn(J, T, D, device=dev)
    msgs = torch.randn(J, T, D, device=dev)
    wn = 1 + 0.1 * torch.randn(J, D, device=dev)
    wdown = torch.stack([c.down_weight.data for c in ctrls]); wg = torch.stack([c.gain_weight.data for c in ctrls])
    wf = torch.stack([c.forget_weight.data for c in ctrls])
    for local in (False, True):
        dest = mixed[:1].expand(J, T, D) if local else mixed
        ref = torch.stack([_rms_norm(ctrls[j](dest[j], msgs[j], 1e-6), wn[0] if local else wn[j], 1e-6) for j in range(J)])
        out = ctrl_blend_norm_op(dest, msgs, wdown, wg, wf, wn[:1] if local else wn, 1e-6, 1e-6)
        chk(f"ctrl T={T} local={local}", out, ref, 1e-2)

# timing
blocks = [torch.randn(1, D, device=dev) for _ in range(3)]; cur = torch.randn(1, D, device=dev)
qry = torch.randn(3, D, device=dev); wn = torch.randn(3, D, device=dev)
def bench(f, n=50):
    for _ in range(3): f()
    torch.xpu.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.time() - t) / n * 1e3
print("mix+norm triton %.3f ms (eager mix+3 norms %.3f ms)" % (
    bench(lambda: mix_op(blocks, cur, qry, norm_weight=wn)),
    bench(lambda: [_rms_norm(x, wn[j], 1e-6) for j, x in enumerate(attn_res_mix(blocks, cur, qry))])))
mixed = torch.randn(J, 1, D, device=dev); msgs = torch.randn(J, 1, D, device=dev)
print("ctrl x3 triton %.3f ms (eager %.3f ms)" % (
    bench(lambda: ctrl_blend_norm_op(mixed, msgs, wdown, wg, wf, wn, 1e-6, 1e-6)),
    bench(lambda: [_rms_norm(ctrls[j](mixed[j], msgs[j], 1e-6), wn[j], 1e-6) for j in range(J)])))
print("ALL PASS" if ok else "SOME FAILED")
