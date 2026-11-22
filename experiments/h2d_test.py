import time, torch
n = 256 * 1024 * 1024
for dev in ("xpu:0",):
    a = torch.empty(n, dtype=torch.uint8, device=dev)
    h = torch.empty(n, dtype=torch.uint8).pin_memory()
    def bw(fn, it=10):
        fn(); torch.xpu.synchronize()
        t = time.time()
        for _ in range(it): fn()
        torch.xpu.synchronize(); return n * it / (time.time() - t) / 1e9
    print(dev, "H2D %.1f GB/s | D2H %.1f GB/s" % (bw(lambda: a.copy_(h)), bw(lambda: h.copy_(a))))
