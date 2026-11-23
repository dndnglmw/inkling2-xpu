import sys

import torch

M, K, N = 8, int(sys.argv[1]), 2304
g = torch.Generator().manual_seed(0)
A = torch.randn(M, K, generator=g).bfloat16().xpu()
W = torch.randn(K, N, generator=g).bfloat16().xpu()
fb = (A @ W).cpu()
fr = torch.cat([A[i:i + 1] @ W for i in range(M)]).cpu()

tb, tr = [], []
for it in range(200):
    b = (A @ W).cpu()
    r = torch.cat([A[i:i + 1] @ W for i in range(M)]).cpu()
    tb.append(1 if (b != fb).any() else 0)
    tr.append(1 if (r != fr).any() else 0)

print("batched timeline:", "".join(map(str, tb)))
print("rows    timeline:", "".join(map(str, tr)))
print("batched flips:", sum(tb), "| first-flip iter:", next((i for i, v in enumerate(tb) if v), None),
      "| last-flip iter:", max((i for i, v in enumerate(tb) if v), default=None))
print("rows    flips:", sum(tr), "| first-flip iter:", next((i for i, v in enumerate(tr) if v), None),
      "| last-flip iter:", max((i for i, v in enumerate(tr) if v), default=None))
