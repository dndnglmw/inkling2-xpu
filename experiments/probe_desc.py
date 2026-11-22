import torch, triton, triton.language as tl

@triton.jit
def k_desc(X, Y, stride_x: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    desc = tl.make_tensor_descriptor(X, shape=[BN, 2 * D], strides=[stride_x, 1],
                                     block_shape=[BN, D])
    a = desc.load([0, 0])
    b = desc.load([0, D])
    tl.store(Y + tl.arange(0, BN)[:, None] * D + tl.arange(0, D)[None, :], a + b)

BN, D = 64, 128
x = torch.randn(BN, 2 * D, device="xpu").to(torch.bfloat16)
y = torch.zeros(BN, D, device="xpu").to(torch.bfloat16)

def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="xpu", dtype=torch.int8)
triton.set_allocator(alloc_fn)

k_desc[(1,)](x, y, stride_x=2 * D, BN=BN, D=D)
torch.xpu.synchronize()
ref = (x[:, :D].float() + x[:, D:].float()).to(torch.bfloat16)
print("descriptor load OK, max err:", (y.float() - ref.float()).abs().max().item())