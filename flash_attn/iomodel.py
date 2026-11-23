"""HBM traffic model: standard attention versus FlashAttention (forward pass).

Standard attention writes S = QK^T to HBM, reads it for softmax, writes P, reads P for PV:
    bytes ~ (3 N d [read Q,K,V] + N d [write O]) * b  +  4 N^2 * b
FlashAttention reads Q once, and streams all of K and V once per query block (N / BLOCK_M times):
    bytes ~ (N d [Q] + 2 N d * ceil(N / BLOCK_M) [K, V] + N d [O]) * b
(b = bytes per element). Both per (batch, head). These are model estimates, not measurements.
"""
import math


def standard_bytes(n, d, elem=2):
    return (4 * n * d + 4 * n * n) * elem


def flash_bytes(n, d, block_m=128, elem=2):
    return (2 * n * d + 2 * n * d * math.ceil(n / block_m)) * elem


def table(ns=(1024, 2048, 4096, 8192, 16384), d=64, block_m=128):
    rows = []
    for n in ns:
        s, f = standard_bytes(n, d), flash_bytes(n, d, block_m)
        rows.append((n, s / 2**20, f / 2**20, s / f))
    return rows


if __name__ == "__main__":
    print("N, standard MiB, flash MiB, ratio  (per batch*head, d=64, fp16, BLOCK_M=128)")
    for n, s, f, r in table():
        print(f"{n:>6} {s:>10.1f} {f:>10.1f} {r:>6.1f}x")
