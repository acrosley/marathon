"""The fused Triton re-rotate-and-scatter must agree with the torch path it replaces.

The kernel is a pure optimisation of :func:`marathon.shift_kernels.scatter_shifted_torch`
(itself lifted verbatim out of the connector), so the only thing worth testing is that
the two write the same bytes. bf16 output makes that essentially exact: the fp32
arithmetic is the same expression in the same order, and any last-bit fp32 difference
from Triton's fma contraction disappears in the bf16 rounding — the tolerance below is
one bf16 ulp, and in practice the outputs are bit-identical.

Covered: both paged layouts, δ of both signs including the large shifts a grow-edit
produces, δ = 0 (plain scatter), a ragged token count that does not fill a block, and
scattered non-contiguous destination blocks (a real block table is not sorted).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from marathon import shift_kernels  # noqa: E402
from marathon.kvshift import rerotate_keys  # noqa: E402

DELTAS = [-3000, -4, 0, 4, 186, 10000]
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")


def _inv_freq(rotary: int = 128, theta: float = 1e6) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary))


def test_tables_match_rerotate_keys():
    """The δ tables encode exactly ``kvshift.rerotate_keys`` (CPU, no Triton needed)."""
    inv = _inv_freq()
    k = torch.randn(2, 8, 5, 128, dtype=torch.float32)
    for delta in DELTAS:
        sh = shift_kernels.rope_shift(delta, 128, inv, "cpu")
        want = rerotate_keys(k, delta, inv)
        got = k * sh.emb_cos + torch.cat((-k[..., 64:], k[..., :64]), -1) * sh.emb_sin
        assert torch.allclose(got, want, atol=1e-6), delta
        # the kernel's branchless form: out[i] = k[i]*cos[i] + k[partner[i]]*sgn[i]
        fused = k * sh.cos + k[..., sh.partner.long()] * sh.sgn
        assert torch.allclose(fused, want, atol=1e-6), delta


@cuda
@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("hnd", [True, False])
@pytest.mark.parametrize("n", [64, 37])
def test_triton_matches_torch(delta: int, hnd: bool, n: int):
    dev = "cuda"
    heads, d, bs, nblocks = 8, 128, 16, 64
    torch.manual_seed(delta * 7 + n)
    src = torch.randn(n, heads, 2 * d, dtype=torch.bfloat16, device=dev)
    shape = (nblocks, heads, bs, 2 * d) if hnd else (nblocks, bs, heads, 2 * d)
    ref = torch.randn(shape, dtype=torch.bfloat16, device=dev)
    got = ref.clone()

    # a real block table is a scattered permutation, and the run need not start at a
    # block boundary — both are cases the connector actually hands the kernel
    table = torch.randperm(nblocks, device=dev)[: -(-n // bs) + 1]
    pos = torch.arange(3, 3 + n, device=dev)
    slots = table[pos // bs] * bs + pos % bs

    shift = shift_kernels.rope_shift(delta, d, _inv_freq(), dev)
    shift_kernels.scatter_shifted_torch(src, ref, slots, bs, hnd, shift)
    shift_kernels.scatter_shifted(src, got, slots, bs, hnd, shift)
    assert shift_kernels.use_triton(got), "expected the Triton path on CUDA"

    # Almost every element is bit-identical; the handful that are not come from Triton
    # contracting ``k*cos + kp*sgn`` into an fma, which differs from torch's separate
    # multiplies in the last fp32 bit. Where the rotation nearly cancels, that last bit
    # is several bf16 ulp *of a value near zero*, so the honest bound is on absolute
    # error against the input scale, not on ulp of the output.
    bits = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    err = (got.float() - ref.float()).abs().max()
    assert err <= 2**-7 * float(src.abs().max()), f"delta={delta} hnd={hnd}: {float(err)}"
    assert float((bits > 1).float().mean()) < 1e-3, "more than 1 in 1000 beyond 1 ulp"


@cuda
def test_zero_length_load_is_a_noop():
    dev = "cuda"
    kv = torch.randn(4, 8, 16, 256, dtype=torch.bfloat16, device=dev)
    before = kv.clone()
    src = torch.empty(0, 8, 256, dtype=torch.bfloat16, device=dev)
    slots = torch.empty(0, dtype=torch.int64, device=dev)
    shift = shift_kernels.rope_shift(4, 128, _inv_freq(), dev)
    shift_kernels.scatter_shifted(src, kv, slots, 16, True, shift)
    assert torch.equal(kv, before)
