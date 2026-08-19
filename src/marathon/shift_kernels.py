"""Fused re-rotate-and-scatter for the shift connector: RoPE δ + paged write in one pass.

:mod:`marathon.vllm_shift_connector` reuses a suffix's KV by copying it out of the
session store into vLLM's paged cache with the K half rotated by δ positions. Done in
torch that is five separate passes over the same bytes — slice K, upcast to fp32, build
``rotate_half`` (a ``cat``, so a full extra allocation), fuse, downcast, ``cat`` K and V
back together, then an advanced-indexing scatter — and the 2026-08-19 length-sweep
measured the result at 22–68 GB/s, ~5 µs per reused token, which is 41% of a 30k edit
turn. It is a pure memory-movement problem being run at a tenth of memcpy speed.

This module does the whole thing in one Triton pass per layer: read each source token's
``[K|V]`` row, rotate K in-register in fp32, write it straight to its destination block
slot in the paged layout. Every byte is read once and written once.

The rotation is folded into two tables so the kernel needs no branches. ``rotate_half``
pairs ``(i, i + d/2)`` with a sign flip on the first half, so with

    partner[i] = i + h  (i < h),  i - h  (i >= h)      h = rotary/2
    sgn[i]     = -sin(δ·inv_freq[i mod h])  for i < h, else +sin(...)

the update is exactly ``out[i] = k[i]·cos[i] + k[partner[i]]·sgn[i]`` — the same
arithmetic as :func:`marathon.kvshift.rerotate_keys`, in the same order, in fp32. Any
non-rotary tail of the head dimension (``partial_rotary_factor < 1``) gets
``cos = 1, sgn = 0`` and passes through untouched.

Both paged layouts are handled by passing strides rather than branching on layout:
HND ``[blocks, heads, block_size, 2*head_size]`` and NHD ``[blocks, block_size, heads,
2*head_size]`` differ only in which stride multiplies the head and which the in-block
offset. :func:`scatter_shifted` dispatches to :func:`scatter_shifted_torch` — the
original implementation, kept as the reference the unit test compares against — when
Triton or CUDA is unavailable, or when ``MARATHON_NO_TRITON`` is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

try:  # Triton ships with torch, but not on every platform (and not on CPU-only boxes).
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only on Triton-less installs
    HAVE_TRITON = False


@dataclass(frozen=True)
class RopeShift:
    """Precomputed per-δ rotation tables, on the device the kernel will run on.

    ``cos``/``sgn`` are fp32 vectors of length ``head_size`` and ``partner`` an int32
    permutation of the same length; see the module docstring for the identity they
    encode. Built once per (δ, device) and cached by the caller — δ is constant for a
    whole load, so this is 3 × 128 floats of setup for megabytes of copy.
    """

    delta: int
    cos: torch.Tensor
    sgn: torch.Tensor
    partner: torch.Tensor
    emb_cos: torch.Tensor  # cat(ang, ang).cos(), the torch path's broadcast table
    emb_sin: torch.Tensor


def rope_shift(delta: int, head_size: int, inv_freq: torch.Tensor, device) -> RopeShift:
    """Tables that move RoPE'd keys ``delta`` positions, for one head dimension."""
    rotary = 2 * int(inv_freq.numel())
    assert rotary <= head_size, f"rotary {rotary} > head_size {head_size}"
    half = rotary // 2
    ang = float(delta) * inv_freq.to(device=device, dtype=torch.float32)
    cos = torch.ones(head_size, dtype=torch.float32, device=device)
    sgn = torch.zeros(head_size, dtype=torch.float32, device=device)
    cos[:rotary] = torch.cat((ang, ang)).cos()
    sin = torch.cat((ang, ang)).sin()
    sgn[:half] = -sin[:half]
    sgn[half:rotary] = sin[half:rotary]
    partner = torch.arange(head_size, dtype=torch.int32, device=device)
    partner[:half] += half
    partner[half:rotary] -= half
    emb = torch.cat((ang, ang))
    return RopeShift(int(delta), cos, sgn, partner, emb.cos(), emb.sin())


if HAVE_TRITON:

    @triton.jit
    def _shift_scatter_kernel(
        src_ptr,  # [n, heads, 2*D] source rows out of the session store
        dst_ptr,  # paged KV cache, addressed through the strides below
        slots_ptr,  # [n] int64 flat slot index of each destination token
        cos_ptr,
        sgn_ptr,
        partner_ptr,  # [D] fp32/fp32/int32 rotation tables
        n_tokens,
        block_size,
        src_tok_stride,
        src_head_stride,
        dst_blk_stride,
        dst_head_stride,
        dst_off_stride,
        ROTATE: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        head = tl.program_id(1)
        toks = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        live = toks < n_tokens

        slot = tl.load(slots_ptr + toks, mask=live, other=0)
        src = src_ptr + toks[:, None] * src_tok_stride + head * src_head_stride
        dst = (
            dst_ptr
            + (slot // block_size)[:, None] * dst_blk_stride
            + (slot % block_size)[:, None] * dst_off_stride
            + head * dst_head_stride
        )

        d = tl.arange(0, D)
        m = live[:, None]
        k = tl.load(src + d[None, :], mask=m, other=0.0).to(tl.float32)
        if ROTATE:
            partner = tl.load(partner_ptr + d)
            kp = tl.load(src + partner[None, :], mask=m, other=0.0).to(tl.float32)
            k = k * tl.load(cos_ptr + d)[None, :] + kp * tl.load(sgn_ptr + d)[None, :]
        tl.store(dst + d[None, :], k.to(dst_ptr.dtype.element_ty), mask=m)

        # V carries no position: straight copy of the second half of the row.
        v = tl.load(src + D + d[None, :], mask=m, other=0.0)
        tl.store(dst + D + d[None, :], v, mask=m)


def _paged_strides(kv: torch.Tensor, hnd: bool) -> tuple[int, int, int]:
    """(block, head, in-block-offset) strides of a fused 4-D paged KV tensor."""
    s = kv.stride()
    return (s[0], s[1], s[2]) if hnd else (s[0], s[2], s[1])


def scatter_shifted_torch(
    src: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    hnd: bool,
    shift: RopeShift | None,
) -> None:
    """Reference path: the original torch implementation, used when Triton is absent."""
    d = src.shape[-1] // 2
    k = src[..., :d]
    if shift is not None and shift.delta:
        kf = k.to(torch.float32)
        half = kf.shape[-1] // 2
        rot = torch.cat((-kf[..., half:], kf[..., :half]), dim=-1)
        k = (kf * shift.emb_cos + rot * shift.emb_sin).to(kv.dtype)
    blk, off = slots // block_size, slots % block_size
    index = (blk, slice(None), off) if hnd else (blk, off)
    kv[index] = torch.cat((k, src[..., d:]), dim=-1)


def use_triton(kv: torch.Tensor) -> bool:
    return HAVE_TRITON and kv.is_cuda and not os.environ.get("MARATHON_NO_TRITON")


def scatter_shifted(
    src: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    hnd: bool,
    shift: RopeShift | None,
) -> None:
    """Write ``src`` (``[n, heads, 2*head_size]``) into ``kv``'s slots, K rotated by δ.

    ``slots`` are flat destination slot indices (block ``slot // block_size``, offset
    ``slot % block_size``); ``hnd`` selects the paged layout. ``shift`` of ``None`` (or
    δ = 0) is a plain scatter.
    """
    if not use_triton(kv):
        return scatter_shifted_torch(src, kv, slots, block_size, hnd, shift)
    n, heads, two_d = src.shape
    d = two_d // 2
    assert kv.shape[-1] == two_d, f"src row {two_d} != kv row {kv.shape[-1]}"
    if n == 0:
        return
    rotate = shift is not None and shift.delta != 0
    zero = src.new_empty(0, dtype=torch.float32)
    blk_s, head_s, off_s = _paged_strides(kv, hnd)
    block_t = 16
    grid = (triton.cdiv(n, block_t), heads)
    _shift_scatter_kernel[grid](
        src,
        kv,
        slots,
        shift.cos if rotate else zero,
        shift.sgn if rotate else zero,
        shift.partner if rotate else zero,
        n,
        block_size,
        src.stride(0),
        src.stride(1),
        blk_s,
        head_s,
        off_s,
        ROTATE=rotate,
        D=d,
        BLOCK_T=block_t,
        num_warps=4,
    )


def warmup(kv: torch.Tensor, block_size: int, hnd: bool) -> None:
    """Compile the kernel against ``kv``'s signature before the first real load.

    An edit turn issues exactly one load, so a cold Triton JIT is charged in full to the
    measurement it is supposed to speed up (it was: 216 ms for a 4.6 GB copy the kernel
    itself does in 6 ms). This runs every specialisation the real load can hit over a
    dummy in scratch memory shaped like the real cache: rotating and not, and both a
    16-divisible and a ragged token count — Triton specialises on divisibility, so a
    one-token warmup would recompile anyway when the real 29,232-token load arrives.
    """
    if not use_triton(kv):
        return
    heads, two_d = kv.shape[1 if hnd else 2], kv.shape[3]
    n = 4 * block_size
    scratch = torch.zeros(
        (4, heads, block_size, two_d) if hnd else (4, block_size, heads, two_d),
        dtype=kv.dtype,
        device=kv.device,
    )
    src = torch.zeros(n, heads, two_d, dtype=kv.dtype, device=kv.device)
    slot = torch.arange(n, dtype=torch.int64, device=kv.device)
    inv = 1.0 / (10000.0 ** (torch.arange(0, two_d // 2, 2, dtype=torch.float32) / (two_d // 2)))
    for shift in (None, rope_shift(1, two_d // 2, inv, kv.device)):
        for count in (n, n - 3):  # aligned and ragged: two Triton specialisations
            scatter_shifted(src[:count], scratch, slot[:count], block_size, hnd, shift)
    torch.cuda.synchronize()
