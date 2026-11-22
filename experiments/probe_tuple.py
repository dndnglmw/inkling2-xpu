import torch, triton, triton.language as tl

@triton.jit
def probe(X, G: tl.constexpr, BM: tl.constexpr, NBLK: tl.constexpr):
    m = ()
    acc = ()
    for g in tl.static_range(G):
        m = m + (tl.full([BM], -1.0, tl.float32) + g,)
        acc = acc + (tl.zeros([BM, 8], tl.float32) + g,)
    for i in range(NBLK):
        s = tl.zeros([BM, 8], tl.float32) + i
        for g in tl.static_range(G):
            mnew = tl.maximum(m[g], tl.max(s, 1))
            p = tl.exp2(s - mnew[:, None])
            m = m[:g] + (mnew,) + m[g + 1:]
            acc = acc[:g] + (acc[g] + tl.dot(p.to(tl.bfloat16), tl.full([8, 8], 1.0, tl.bfloat16)),) + acc[g + 1:]
    x = m[0]
    if G > 1:
        x = x + tl.sum(acc[1], 1)
    tl.store(X + tl.arange(0, BM), x)

x = torch.zeros(64, device="xpu")
probe[(1,)](x, G=3, BM=64, NBLK=4)
torch.xpu.synchronize()
print("static_range tuple OK", x[:3].tolist())