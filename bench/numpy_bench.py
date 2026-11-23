"""CPU benchmark of the algorithm (NumPy): peak memory and time, naive versus blocked.
This measures the MEMORY effect of never materialising N x N scores. Speed on a GPU is a
different question; see bench/gpu_bench.py."""
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_attn import attention_blocked, attention_naive  # noqa: E402


def measure(fn):
    tracemalloc.start()
    t = time.perf_counter()
    fn()
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return dt, peak


if __name__ == "__main__":
    r = np.random.default_rng(0)
    print(f"{'N':>6} {'naive MiB':>10} {'blocked MiB':>12} {'naive s':>8} {'blocked s':>10}   (1 head, d=64, float32, blocks 128x128)")
    for n in (1024, 2048, 4096, 8192):
        q, k, v = (r.standard_normal((1, n, 64)).astype(np.float32) for _ in range(3))
        tn, mn = measure(lambda: attention_naive(q, k, v))
        tb, mb = measure(lambda: attention_blocked(q, k, v, block_m=128, block_n=128))
        print(f"{n:>6} {mn:>10.1f} {mb:>12.1f} {tn:>8.3f} {tb:>10.3f}")
