import hashlib
import json
import sys

import torch

M, K, N = 8, int(sys.argv[1]), 2304
g = torch.Generator().manual_seed(0)
A = torch.randn(M, K, generator=g).bfloat16().xpu()
W = torch.randn(K, N, generator=g).bfloat16().xpu()
h = lambda t: hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]

B32 = lambda: torch.bmm(A.unsqueeze(0), W.t().unsqueeze(0), out_dtype=torch.float32)[0].cpu()
R_bmm32 = lambda: torch.cat([torch.bmm(A[i:i + 1].unsqueeze(0), W.t().unsqueeze(0), out_dtype=torch.float32)[0]
                             for i in range(M)]).cpu()

fb = B32()
res = {"dig_batched32": h(fb)}
for n, f in (("R_bmm32", R_bmm32),):
    o = f()
    flips = sum(1 for _ in range(50) if (f() != fb).any())
    res[n] = {"digest": h(o), "bit_identical_to_batched32": bool((o == fb).all()), "flips_in_50": flips}
print(json.dumps(res))
