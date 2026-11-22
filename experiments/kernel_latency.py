import time, torch
import triton, triton.language as tl
dev = torch.device("xpu")
@triton.jit
def _inc(X, N, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    x = tl.load(X + o, mask=o < N)
    tl.store(X + o, x + 1.0, mask=o < N)
x = torch.zeros(1024, device=dev)
def run_triton(n):
    for _ in range(n): _inc[(1,)](x, 1024, BLOCK=1024)
def run_torch(n):
    for _ in range(n): x.add_(1.0)
for name, fn in (("triton tiny kernel", run_triton), ("torch add_", run_torch)):
    fn(10); torch.xpu.synchronize()
    # eager
    t = time.time(); fn(1000); torch.xpu.synchronize(); eager = (time.time() - t) / 1000 * 1e6
    # graph
    g = torch.xpu.XPUGraph()
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        fn(5); torch.xpu.synchronize()
        with torch.xpu.graph(g, stream=s):
            fn(1000)
    torch.xpu.synchronize()
    g.replay(); torch.xpu.synchronize()
    t = time.time()
    for _ in range(5): g.replay()
    torch.xpu.synchronize(); graph = (time.time() - t) / 5 / 1000 * 1e6
    print(f"{name:20s}: eager {eager:6.1f} us/kernel | XPU-graph replay {graph:6.1f} us/kernel (dependent chain)")
