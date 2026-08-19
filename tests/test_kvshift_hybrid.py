"""CPU tests for the pure logic in ``kvshift_hybrid``: partial rotary, and the cost model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marathon.kvshift import rerotate_keys, rotate_half  # noqa: E402
from marathon.kvshift_hybrid import CostModel, HybridPolicy, rerotate_keys_partial  # noqa: E402

HEAD_DIM = 256
ROT = 64  # partial_rotary_factor 0.25, as in Qwen3.5


def _inv_freq(rot: int = ROT, base: float = 1e7) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))


def _rope(k: torch.Tensor, start: int, inv: torch.Tensor) -> torch.Tensor:
    """Apply RoPE at positions ``start..`` to the leading ``2*len(inv)`` dims only."""
    rot = 2 * inv.shape[0]
    pos = torch.arange(start, start + k.shape[-2], dtype=inv.dtype)
    ang = pos[:, None] * inv
    emb = torch.cat((ang, ang), dim=-1)
    head = k[..., :rot] * emb.cos() + rotate_half(k[..., :rot]) * emb.sin()
    return torch.cat([head, k[..., rot:]], dim=-1)


def test_partial_rerotate_matches_recompute_at_the_shifted_position():
    inv = _inv_freq()
    k = torch.randn(1, 4, 9, HEAD_DIM, dtype=torch.float64)
    got = rerotate_keys_partial(_rope(k, 100, inv), 37, inv)
    # rerotate_keys builds its cos/sin in fp32 by design, so fp32 accuracy is the bar
    assert torch.allclose(got, _rope(k, 137, inv), atol=1e-5)


def test_partial_rerotate_leaves_the_unrotated_tail_untouched():
    """The tail dims never carried a position; moving them would be a bug."""
    inv = _inv_freq()
    k = torch.randn(1, 4, 9, HEAD_DIM, dtype=torch.float64)
    got = rerotate_keys_partial(k, 37, inv)
    assert torch.equal(got[..., ROT:], k[..., ROT:])
    assert not torch.equal(got[..., :ROT], k[..., :ROT])


def test_partial_rerotate_degenerates_to_the_full_one_when_rotary_is_full():
    inv = _inv_freq(rot=HEAD_DIM)
    k = torch.randn(1, 4, 9, HEAD_DIM, dtype=torch.float64)
    assert torch.equal(rerotate_keys_partial(k, 5, inv), rerotate_keys(k, 5, inv))


def test_rerotate_by_zero_is_identity():
    inv = _inv_freq()
    k = torch.randn(1, 2, 3, HEAD_DIM)
    assert rerotate_keys_partial(k, 0, inv) is k


def test_cost_model_fractions_bracket_correctly():
    """Stale < mix < hidden < full, and a full recompute costs exactly 1.0."""
    cost = CostModel(full_token=1000, replay_hidden=150, replay_mix=14, n_linear=24, n_attn=8)
    total, fresh = 1000, 50
    replayed = total - fresh
    stale = cost.frac(fresh, 0, total, "none")
    mix = cost.frac(fresh, replayed, total, "mix")
    hidden = cost.frac(fresh, replayed, total, "hidden")
    assert stale == pytest.approx(0.05)
    assert stale < mix < hidden < 1.0
    assert cost.frac(total, 0, total, "none") == pytest.approx(1.0)
    # replay cost is exactly the per-token proxy times the replayed count
    assert mix == pytest.approx((50 * 1000 + 950 * 14) / (1000 * 1000))


def test_policy_labels_are_distinct_and_readable():
    labels = [
        HybridPolicy(linear="stale", rerotate=False).label(),
        HybridPolicy(linear="stale").label(),
        HybridPolicy(linear="mix").label(),
        HybridPolicy(linear="hidden").label(),
        HybridPolicy(linear="hidden", first_m=256).label(),
    ]
    assert labels == [
        "no-rerotate",
        "stale-state",
        "replay-mix",
        "replay-hidden",
        "replay-hidden+first256",
    ]
    assert len(set(labels)) == len(labels)


def test_policy_mode_maps_to_the_cost_model_keys():
    for pol, mode in [("stale", "none"), ("hidden", "hidden"), ("mix", "mix")]:
        assert HybridPolicy(linear=pol).mode == mode
