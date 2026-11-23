"""Reference attention in NumPy.

attention_naive    materialises the full N x N score matrix (what PyTorch does without fused kernels).
attention_blocked  is the FlashAttention algorithm written with NumPy on blocks: it walks the keys
                   in blocks of BLOCK_N, keeps a running max m, running sum l and an output
                   accumulator per query row, and never stores more than BLOCK_M x BLOCK_N scores.
                   It mirrors flash_attn/triton_fwd.py line by line, so it documents and checks the kernel.
Shapes: q, k, v are [..., N, D] (any leading batch/head dimensions).
"""
import numpy as np


def attention_naive(q, k, v, causal=False, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = (q @ np.swapaxes(k, -1, -2)) * scale                      # [..., N, N]: O(N^2) memory
    if causal:
        n = q.shape[-2]
        s = np.where(np.tril(np.ones((n, k.shape[-2]), dtype=bool)), s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def attention_blocked(q, k, v, causal=False, scale=None, block_m=64, block_n=64, return_lse=False):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    n_q, n_k = q.shape[-2], k.shape[-2]
    out = np.empty(q.shape[:-1] + (v.shape[-1],), dtype=np.result_type(q, v))
    lse = np.empty(q.shape[:-1], dtype=out.dtype)
    for start_m in range(0, n_q, block_m):                         # one Triton program per query block
        qb = q[..., start_m:start_m + block_m, :]
        rows = np.arange(start_m, start_m + qb.shape[-2])
        m_i = np.full(qb.shape[:-1], -np.inf, dtype=out.dtype)     # running max of each row
        l_i = np.zeros(qb.shape[:-1], dtype=out.dtype)             # running sum of exp(s - m_i)
        acc = np.zeros(qb.shape[:-1] + (v.shape[-1],), dtype=out.dtype)
        hi = min(start_m + qb.shape[-2], n_k) if causal else n_k   # causal: later key blocks are fully masked
        for start_n in range(0, hi, block_n):
            kb = k[..., start_n:start_n + block_n, :]
            vb = v[..., start_n:start_n + block_n, :]
            s = (qb @ np.swapaxes(kb, -1, -2)) * scale             # [..., BLOCK_M, BLOCK_N] only
            if causal:
                cols = np.arange(start_n, start_n + kb.shape[-2])
                s = np.where(rows[:, None] >= cols[None, :], s, -np.inf)
            m_new = np.maximum(m_i, s.max(axis=-1))
            p = np.exp(s - m_new[..., None])
            alpha = np.exp(m_i - m_new)                            # rescale what we summed so far
            l_i = l_i * alpha + p.sum(axis=-1)
            acc = acc * alpha[..., None] + p @ vb
            m_i = m_new
        out[..., start_m:start_m + qb.shape[-2], :] = acc / l_i[..., None]
        lse[..., start_m:start_m + qb.shape[-2]] = m_i + np.log(l_i)   # saved for the backward pass
    return (out, lse) if return_lse else out
