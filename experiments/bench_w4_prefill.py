import time, torch, torch.nn.functional as F
import vllm._xpu_ops  # noqa: registers fakes; ops come from vllm_xpu_kernels
import vllm_xpu_kernels  # noqa
dev = torch.device("xpu")
ops = torch.ops._xpu_C
print("ops present: int4_gemm_w4a16", hasattr(ops, "int4_gemm_w4a16"), "| fp8_gemm_w8a16", hasattr(ops, "fp8_gemm_w8a16"))
def bench(f, n=30):
    for _ in range(10): f()
    torch.xpu.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.time() - t) / n * 1e6
G = 128
shapes = {"qkvr (TP2)": (4300, 2304), "wo (TP2)": (2304, 1536)}
print(f"{'layer':14s} {'M':>2s} | {'bf16 us':>8s} {'GB/s':>6s} | {'w4a16 us':>9s} {'speedup':>7s} | {'fp8 us':>7s} {'speedup':>7s}")
for name, (N, K) in shapes.items():
    W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    qw = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)      # packed 8 x uint4 along K
    sc = (torch.rand(K // G, N, device=dev) * 0.01).to(torch.bfloat16)
    zp = torch.tensor([8], dtype=torch.int8, device=dev)
    w8 = W.to(torch.float8_e4m3fn).t()                                                       # [K, N]
    s8 = (torch.rand(1, N, device=dev) * 0.01).float()
    for M in (512, 2048, 8192):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        a = bench(lambda: F.linear(x, W))
        gbs = N * K * 2 / (a * 1e-6) / 1e9
        try:
            b = bench(lambda: ops.int4_gemm_w4a16(x, qw.t(), None, sc, zp, G, None))
            out = ops.int4_gemm_w4a16(x, qw.t(), None, sc, zp, G, None); ok4 = bool(torch.isfinite(out).all())
        except Exception as e:
            b = float("nan"); ok4 = False; err4 = str(e)[:80]
        try:
            c = bench(lambda: ops.fp8_gemm_w8a16(x, w8, s8, None))
        except Exception as e:
            c = float("nan")
        print(f"{name:14s} {M:2d} | {a:8.1f} {gbs:6.0f} | {b:9.1f} {a/b:6.2f}x | {c:7.1f} {a/c:6.2f}x {'' if ok4 else '(w4 nonfinite/err)'}")
