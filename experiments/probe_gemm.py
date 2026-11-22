import torch
dev = torch.device("xpu")
for n in (2048, 4096, 8192):
    a = torch.randn(n, n, device=dev).to(torch.bfloat16)
    b = torch.randn(n, n, device=dev).to(torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.xpu.synchronize()
    import time
    t = time.time()
    for _ in range(10):
        c = a @ b
    torch.xpu.synchronize()
    dt = (time.time() - t) / 10
    print(f"{n}x{n}x{n} bf16: {dt*1e3:.2f} ms  {2*n**3/dt/1e12:.1f} TFLOP/s")