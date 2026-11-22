import time, torch
assert torch.xpu.device_count() >= 2
n = 256 * 1024 * 1024  # 256 MB
a = torch.empty(n, dtype=torch.uint8, device="xpu:0").fill_(1)
b = torch.empty(n, dtype=torch.uint8, device="xpu:1")
def bw(fn, it=10):
    fn(); torch.xpu.synchronize(0); torch.xpu.synchronize(1)
    t = time.time()
    for _ in range(it): fn()
    torch.xpu.synchronize(0); torch.xpu.synchronize(1)
    return n * it / (time.time() - t) / 1e9
print("direct device copy xpu:0 -> xpu:1 : %.1f GB/s" % bw(lambda: b.copy_(a)))
h = torch.empty(n, dtype=torch.uint8).pin_memory()
def staged():
    h.copy_(a); b.copy_(h)
print("host-staged (via pinned RAM)      : %.1f GB/s (both hops)" % bw(staged))
print("D2H only %.1f GB/s | H2D only %.1f GB/s" % (bw(lambda: h.copy_(a)), bw(lambda: b.copy_(h))))
try:
    print("can_device_access_peer(0,1):", torch.xpu.can_device_access_peer(0, 1))
except Exception as e:
    print("can_device_access_peer n/a:", type(e).__name__)
