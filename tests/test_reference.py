"""The blocked (FlashAttention) algorithm must equal naive attention. Needs only NumPy."""
import tracemalloc

import numpy as np
import pytest

from flash_attn import attention_blocked, attention_naive


def qkv(n, d=32, heads=2, seed=0, dtype=np.float64):
    r = np.random.default_rng(seed)
    return [r.standard_normal((heads, n, d)).astype(dtype) for _ in range(3)]


@pytest.mark.parametrize("n", [1, 7, 64, 65, 200])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(16, 16), (64, 32), (32, 64)])
def test_blocked_equals_naive(n, causal, blocks):
    q, k, v = qkv(n)
    ref = attention_naive(q, k, v, causal=causal)
    out = attention_blocked(q, k, v, causal=causal, block_m=blocks[0], block_n=blocks[1])
    assert np.max(np.abs(out - ref)) < 1e-12


def test_large_scores_are_stable():
    q, k, v = qkv(100)
    out = attention_blocked(q * 40, k * 40, v, block_m=16, block_n=16)   # scores in the thousands
    assert np.isfinite(out).all()
    assert np.max(np.abs(out - attention_naive(q * 40, k * 40, v))) < 1e-9


def test_logsumexp_matches():
    q, k, v = qkv(90)
    _, lse = attention_blocked(q, k, v, causal=True, return_lse=True, block_m=32, block_n=16)
    s = q @ np.swapaxes(k, -1, -2) * q.shape[-1] ** -0.5
    s = np.where(np.tril(np.ones((90, 90), dtype=bool)), s, -np.inf)
    ref = np.log(np.exp(s - s.max(-1, keepdims=True)).sum(-1)) + s.max(-1)
    assert np.max(np.abs(lse - ref)) < 1e-12


def test_rows_sum_to_one_through_v_of_ones():
    q, k, _ = qkv(77)
    out = attention_blocked(q, k, np.ones((2, 77, 32)), causal=True, block_m=16, block_n=16)
    assert np.allclose(out, 1.0)          # softmax weights of every row sum to 1


def peak_mib(fn):
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2**20


def test_blocked_uses_far_less_memory():
    q, k, v = qkv(2048, d=64, heads=1, dtype=np.float32)
    naive = peak_mib(lambda: attention_naive(q, k, v))
    blocked = peak_mib(lambda: attention_blocked(q, k, v, block_m=64, block_n=64))
    assert blocked * 5 < naive, (naive, blocked)
