"""FlashAttention forward kernel in Triton.

One program computes BLOCK_M query rows of one (batch, head). It loads its Q block once, then
streams K and V in blocks of BLOCK_N through on-chip memory, using the online softmax to keep a
running max (m_i), running sum (l_i) and output accumulator (acc). The N x N score matrix never
exists in GPU memory, so memory is O(N) and HBM traffic drops sharply.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, O, LSE,
              stride_qb, stride_qh, stride_qm, stride_qd,
              stride_kb, stride_kh, stride_kn, stride_kd,
              stride_vb, stride_vh, stride_vn, stride_vd,
              stride_ob, stride_oh, stride_om, stride_od,
              H, N_CTX, sm_scale,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)                       # which query block
    off_bh = tl.program_id(1)                        # which (batch, head)
    off_b, off_h = off_bh // H, off_bh % H
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_CTX, other=0.0)               # [BLOCK_M, D], loaded once

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)   # key blocks after the diagonal are fully masked
    else:
        hi = N_CTX
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=cols[:, None] < N_CTX, other=0.0)             # [BLOCK_N, D]
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=cols[:, None] < N_CTX, other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale          # [BLOCK_M, BLOCK_N] in registers
        valid = cols[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    tl.store(LSE + off_bh * N_CTX + offs_m, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention(q, k, v, causal=False, sm_scale=None, block_m=64, block_n=64, return_lse=False):
    """q, k, v: [batch, heads, seq, head_dim] tensors on the same device. Returns O (and LSE)."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("this kernel expects q, k and v with the same shape [B, H, N, D]")
    b, h, n, d = q.shape
    if d not in (16, 32, 64, 128):
        raise ValueError(f"head_dim must be 16, 32, 64 or 128, got {d}")
    sm_scale = d ** -0.5 if sm_scale is None else sm_scale
    o = torch.empty_like(q)
    lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_m), b * h)
    _attn_fwd[grid](q, k, v, o, lse,
                    *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                    h, n, sm_scale,
                    BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d, CAUSAL=causal)
    return (o, lse) if return_lse else o
