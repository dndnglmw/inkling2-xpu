"""Fused attention prep (q/k/v/r row ops + KV-cache write in one launch) vs rows_op x4 + kv_write: must be bit-identical."""
import sys

import torch

sys.path.insert(0, "/work/vllm")
from vllm.models.inkling2.xpu.triton_prep import kv_write, prep_op, rows_op

dev = torch.device("xpu")
H, HKV, D, DR, BS = 12, 4, 128, 16, 16
ok = True
for T in (1, 4, 8, 37, 300):
    for fp8 in (False, True):
        for use_tau in (False, True):
            torch.manual_seed(T)
            wide = torch.randn(T, 5000, device=dev).to(torch.bfloat16)  # slices below are row-strided like the qkvr split
            q = wide[:, 0:H * D].view(T, H, D)
            r = wide[:, 1600:1600 + H * DR].view(T, H, DR)
            k = wide[:, 2000:2000 + HKV * D].view(T, HKV, D)
            v = wide[:, 3000:3000 + HKV * D].view(T, HKV, D)
            q_scale = torch.randn(H, D, device=dev).to(torch.bfloat16)
            r_scale = torch.randn(H, DR, device=dev).to(torch.bfloat16)
            tau = (torch.rand(T, device=dev) + 0.5) if use_tau else None
            nblk = (T + BS - 1) // BS + 2
            cdt = torch.float8_e4m3fn if fp8 else torch.bfloat16
            cache0 = torch.zeros(nblk, HKV, BS, 2 * D, device=dev, dtype=torch.bfloat16).to(cdt)
            slots = torch.arange(T, device=dev, dtype=torch.int64) + BS
            if T > 3:
                slots[2] = -1  # padded row
            # reference path
            c_ref = cache0.clone()
            q_r = rows_op(q, mode=0, threshold=0.9, scale=q_scale, tau=tau)
            k_r = rows_op(k, mode=0, threshold=0.9)
            v_r = rows_op(v, mode=1, eps=1e-6)
            r_r = rows_op(r, mode=0, threshold=1.3, scale=r_scale, tau=tau)
            kv_write(k_r, v_r, c_ref, slots)
            # fused
            c_new = cache0.clone()
            q_n, r_n = prep_op(q, k, v, r, q_scale=q_scale, r_scale=r_scale, tau=tau, q_thr=0.9, r_thr=1.3, k_thr=0.9, eps=1e-6, cache=c_new, slots=slots)
            same = torch.equal(q_r, q_n) and torch.equal(r_r, r_n) and torch.equal(c_ref.view(torch.uint8) if fp8 else c_ref, c_new.view(torch.uint8) if fp8 else c_new)
            ok &= same
            print(f"{'PASS' if same else 'FAIL'} T={T:3d} fp8={fp8!s:5} tau={use_tau!s:5}")
print("ALL PASS" if ok else "SOME FAILED")
