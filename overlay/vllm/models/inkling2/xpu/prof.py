# SPDX-License-Identifier: Apache-2.0
"""Optional phase timing (INKLING_PROF=1): synchronises around each phase."""
import os
import time
from contextlib import contextmanager

import torch

PROF = os.environ.get("INKLING_PROF") == "1"
TIMES: dict[str, float] = {}


_TP_CALLS: dict[str, int] = {}


def tprof_call(tag: str, n_tokens: int, fn, *args):
    """One-shot torch-profiler capture (device kernel time) of the 40th small call, armed by the flag file
    /work/agents/tprof_<tag>.flag (deleted after the capture). Only active when INKLING_PROF=1."""
    flag = f"/work/agents/tprof_{tag}.flag"
    if not PROF or n_tokens > 16 or not os.path.exists(flag):
        return fn(*args)
    try:
        want = int(open(flag).read().strip() or 0)
    except (OSError, ValueError):
        want = 0
    if want and n_tokens != want:
        return fn(*args)
    c = _TP_CALLS[tag] = _TP_CALLS.get(tag, 0) + 1
    if c < 40:
        return fn(*args)
    _TP_CALLS[tag] = 0
    try:
        os.remove(flag)
    except OSError:
        pass
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.XPU]) as p:
        out = fn(*args)
        torch.xpu.synchronize()
    rows = []
    for e in p.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = e.cuda_time_total
        if t > 0 and not e.key.startswith("aten::"):
            rows.append((e.key, t, e.count))
    rows.sort(key=lambda r: -r[1])
    try:
        print(f"TPROF {tag} positions[:16]={args[0][:16].tolist()}", flush=True)
    except Exception:
        pass
    print(f"TPROF {tag} T={n_tokens}: device total {sum(r[1] for r in rows) / 1000:.2f} ms over {sum(r[2] for r in rows)} kernels", flush=True)
    for k, t, c in rows[:50]:
        print(f"TPROF   {t / 1000:7.3f} ms x{c:<4d} {k[:96]}", flush=True)
    return out


@contextmanager
def ph(name: str):
    if not PROF:
        yield
        return
    torch.xpu.synchronize()
    t = time.perf_counter()
    yield
    torch.xpu.synchronize()
    TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t
