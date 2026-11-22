"""Per-channel FP8 dense projections (INKLING_XPU_FP8_DENSE) vs the bf16 linear."""
import pytest
import torch

from vllm.models.inkling2.xpu.attention import InklingXPUAttention

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="needs XPU")


@pytest.mark.parametrize("n,k", [(5000, 3072), (3072, 2048), (512, 3072)])
@pytest.mark.parametrize("m", [1, 4, 8, 64])
def test_fp8_pack_matches_bf16(n, k, m):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="xpu") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="xpu").to(torch.bfloat16)
    wq, s = InklingXPUAttention._fp8_pack(w)
    assert wq.shape == (k, n) and s.shape == (1, n) and s.dtype == torch.float32
    y = torch.ops._xpu_C.fp8_gemm_w8a16(x, wq, s, None).float()
    ref = torch.nn.functional.linear(x, w).float()
    rel = (y - ref).norm() / ref.norm()
    assert rel < 0.05, rel.item()
