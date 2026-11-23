"""The Triton kernel against PyTorch. On a GPU it runs compiled; without one, in the Triton interpreter."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flash_attn import flash_attention  # noqa: E402

ON_GPU = torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1"
DEVICE = "cuda" if ON_GPU else "cpu"
DTYPES = [torch.float16, torch.bfloat16] if ON_GPU else [torch.float32]


def ref(q, k, v, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return torch.softmax(s, -1) @ v.float()


@pytest.mark.parametrize("n", [16, 50, 128, 200])
@pytest.mark.parametrize("d", [16, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_pytorch(n, d, causal, dtype):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, n, d, device=DEVICE, dtype=dtype) for _ in range(3))
    out = flash_attention(q, k, v, causal=causal, block_m=32, block_n=32)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert (out.float() - ref(q, k, v, causal)).abs().max().item() < tol


def test_lse_matches():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 70, 32, device=DEVICE) for _ in range(3))
    _, lse = flash_attention(q, k, v, causal=True, block_m=32, block_n=32, return_lse=True)
    s = (q @ k.transpose(-1, -2)) * 32 ** -0.5
    s = s.masked_fill(~torch.ones(70, 70, dtype=torch.bool, device=DEVICE).tril(), float("-inf"))
    assert (lse - torch.logsumexp(s, -1)).abs().max().item() < 1e-4


def test_rejects_bad_head_dim():
    q = torch.randn(1, 1, 8, 24, device=DEVICE)
    with pytest.raises(ValueError):
        flash_attention(q, q, q)
