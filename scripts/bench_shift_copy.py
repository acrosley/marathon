"""Micro-benchmark: torch vs fused-Triton re-rotate-and-scatter, Qwen3-14B shapes.

Reproduces exactly what the connector does on an edit turn — for each of 40 layers,
read a run of tokens out of a position-indexed store and scatter them into vLLM's paged
fused KV cache with the K half re-rotated by δ — and reports the achieved bandwidth.
"Bytes" counts the source read once plus the destination write once, which is the
memcpy floor for the operation; the RTX 5090 does ~1.5 TB/s device-to-device.

    python scripts/bench_shift_copy.py [--layers 40] [--delta 186] [--iters 5]
"""

from __future__ import annotations

import argparse
import time

import torch

from marathon import shift_kernels

HEADS, HEAD_SIZE, BLOCK_SIZE = 8, 128, 16  # Qwen3-14B KV geometry, vLLM default paging


def _build(n_tokens: int, layers: int, dev: str):
    """A store slice and a paged KV cache per layer, plus a scattered block table."""
    row = 2 * HEAD_SIZE
    blocks = -(-n_tokens // BLOCK_SIZE) + 8
    src = [
        torch.randn(n_tokens, HEADS, row, dtype=torch.bfloat16, device=dev) for _ in range(layers)
    ]
    kv = [
        torch.zeros(blocks, HEADS, BLOCK_SIZE, row, dtype=torch.bfloat16, device=dev)
        for _ in range(layers)
    ]
    table = torch.randperm(blocks, device=dev)
    pos = torch.arange(n_tokens, device=dev)
    slots = table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
    return src, kv, slots


def _time(fn, src, kv, slots, shift, iters: int) -> float:
    """Milliseconds per full 40-layer pass, best of ``iters`` after a warmup."""
    best = float("inf")
    for i in range(iters + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s, k in zip(src, kv, strict=True):
            fn(s, k, slots, BLOCK_SIZE, True, shift)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if i:  # first pass is warmup (Triton JIT, allocator)
            best = min(best, ms)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--delta", type=int, default=186)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tokens", type=int, nargs="*", default=[4096, 12288, 30720])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark needs a GPU"
    dev = "cuda"
    inv = 1.0 / (1e6 ** (torch.arange(0, HEAD_SIZE, 2, dtype=torch.float32) / HEAD_SIZE))
    shift = shift_kernels.rope_shift(args.delta, HEAD_SIZE, inv, dev)

    print(
        f"{torch.cuda.get_device_name(0)}  {args.layers} layers x {HEADS} kv heads x "
        f"{HEAD_SIZE} dim, bf16, delta={args.delta}, best of {args.iters}"
    )
    print(
        f"{'tokens':>7} {'MB':>7} | {'torch ms':>9} {'GB/s':>6} {'us/tok':>7} | "
        f"{'triton ms':>9} {'GB/s':>6} {'us/tok':>7} | speedup"
    )
    for n in args.tokens:
        src, kv, slots = _build(n, args.layers, dev)
        moved = 2 * sum(s.numel() * s.element_size() for s in src)  # read + write
        t_ms = _time(shift_kernels.scatter_shifted_torch, src, kv, slots, shift, args.iters)
        k_ms = _time(shift_kernels.scatter_shifted, src, kv, slots, shift, args.iters)
        row = [n, moved / 2**20 / 2]
        for ms in (t_ms, k_ms):
            row += [ms, moved / 2**30 / (ms / 1e3), ms * 1e3 / n]
        print(
            "{:7d} {:7.0f} | {:9.2f} {:6.1f} {:7.3f} | {:9.2f} {:6.1f} {:7.3f} | {:5.1f}x".format(
                *row, t_ms / k_ms
            )
        )
        del src, kv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
