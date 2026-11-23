"""GPU benchmark (needs CUDA): this kernel vs naive PyTorch vs torch's fused SDPA.
Usage: python bench/gpu_bench.py [--causal]. Prints ms, TFLOP/s and peak memory."""
import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_attn import flash_attention  # noqa: E402


def naive(q, k, v, causal):
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


def run(fn):
    torch.cuda.reset_peak_memory_stats()
    ms = triton.testing.do_bench(fn, warmup=10, rep=50)
    return ms, torch.cuda.max_memory_allocated() / 2**20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", action="store_true")
    a = ap.parse_args()
    b, h, d = 4, 16, 64
    print(torch.cuda.get_device_name(), "| B=4 H=16 D=64 fp16 | causal =", a.causal)
    print(f"{'N':>6} | {'naive ms':>9} {'TFLOP/s':>8} {'MiB':>7} | {'sdpa ms':>8} {'TFLOP/s':>8} | {'triton ms':>9} {'TFLOP/s':>8} {'MiB':>7}")
    for n in (512, 1024, 2048, 4096, 8192):
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))
        flops = 4 * b * h * n * n * d * (0.5 if a.causal else 1.0)
        tf = lambda ms: flops / ms / 1e9
        try:
            nm, nmem = run(lambda: naive(q, k, v, a.causal))
            naive_s = f"{nm:>9.3f} {tf(nm):>8.1f} {nmem:>7.0f}"
        except torch.cuda.OutOfMemoryError:
            naive_s = f"{'OOM':>9} {'-':>8} {'-':>7}"
            torch.cuda.empty_cache()
        sm, _ = run(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=a.causal))
        fm, fmem = run(lambda: flash_attention(q, k, v, causal=a.causal, block_m=128, block_n=64))
        print(f"{n:>6} | {naive_s} | {sm:>8.3f} {tf(sm):>8.1f} | {fm:>9.3f} {tf(fm):>8.1f} {fmem:>7.0f}")
