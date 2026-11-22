"""Measured dispatch cost per kernel inside an XPU graph replay: N tiny dependent kernels (like the decode step's casts/mixes) captured in a graph.
Reports replay ms for N = 100 / 500 / 2000 / 4000 and the slope (us per kernel) = the per-kernel gap floor on this stack, plus the same N eager."""
import time
import torch

dev = torch.device("xpu")
x = torch.randn(4, 2304, device=dev)
w = torch.randn(4, 2304, device=dev)


def body(n):
    y = x
    for _ in range(n):
        y = torch.add(y, w)  # one tiny kernel, dependent chain (like the decode step)
    return y


def timeit(f, n=10):
    for _ in range(3):
        f()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


res = {}
for n in (100, 500, 2000, 4000):
    body(n); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        out = body(n)
    torch.xpu.synchronize()
    res[n] = (timeit(g.replay), timeit(lambda: body(n)))
    print(f"N={n:5d}: graph replay {res[n][0]:7.3f} ms ({res[n][0] / n * 1000:5.2f} us/kernel) | eager {res[n][1]:7.3f} ms ({res[n][1] / n * 1000:5.2f} us/kernel)", flush=True)
slope = (res[4000][0] - res[500][0]) / 3500 * 1000
print(f"graph slope (500->4000): {slope:.2f} us per tiny kernel")
