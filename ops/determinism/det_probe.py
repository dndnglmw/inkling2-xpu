import hashlib
import json
import sys

import torch

M, K, N = 8, int(sys.argv[1]), 2304
g = torch.Generator().manual_seed(0)
A = torch.randn(M, K, generator=g).bfloat16().xpu()
W = torch.randn(K, N, generator=g).bfloat16().xpu()
h = lambda t: hashlib.sha256(t.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
B = lambda: (A @ W).cpu()
R = lambda: torch.cat([A[i:i + 1] @ W for i in range(M)]).cpu()
fb, fr = B(), R()
fl = {"batched": {}, "rows": {}}
for _ in range(200):
    for n, o, f in (("batched", B(), fb), ("rows", R(), fr)):
        for ix in (o != f).nonzero().tolist():
            fl[n][str(ix)] = fl[n].get(str(ix), 0) + 1
print(json.dumps({"dig_b": h(fb), "dig_r": h(fr),
                  "xlayout": (fb != fr).nonzero().tolist(),
                  "within_proc_flips": fl}))
