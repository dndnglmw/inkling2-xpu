import sys, time, torch
n = 256 * 1024 * 1024
for i in (0, 1):
    dev = f"xpu:{i}"
    a = torch.empty(n, dtype=torch.uint8, device=dev)
    h = torch.empty(n, dtype=torch.uint8).pin_memory()
    def bw(fn, it=8):
        fn(); torch.xpu.synchronize(i)
        t = time.time()
        for _ in range(it): fn()
        torch.xpu.synchronize(i); return n * it / (time.time() - t) / 1e9
    print(f"{dev}: H2D {bw(lambda: a.copy_(h)):5.1f} GB/s | D2H {bw(lambda: h.copy_(a)):5.1f} GB/s")
