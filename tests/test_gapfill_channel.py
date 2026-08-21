"""The connector <-> patched-vLLM hand-off, exercised the way the patch drives it."""

from __future__ import annotations

import numpy as np
import pytest

from marathon import gapfill_channel as ch
from marathon.gapfill import plan_gaps


@pytest.fixture(autouse=True)
def _clean():
    ch.clear()
    yield
    ch.clear()


def test_offer_take_publish_active_release_round_trip():
    p = plan_gaps([{"dst_start": 100, "dst_end": 400, "delta": -5}], 16, 1000)
    ch.offer("r1", p.compute, p.filled_tokens)

    taken = ch.take("r1")
    assert taken is not None
    positions, matched = taken
    assert matched == p.filled_tokens
    assert positions.tolist() == list(p.compute)

    ch.publish("r1", positions, matched)
    assert set(ch.active()) == {"r1"}
    ch.release("r1")
    assert ch.active() == {}


def test_take_consumes_so_a_stale_offer_cannot_be_reused():
    """A plan belongs to one scheduling decision; leaving it would mis-drive the next."""
    ch.offer("r1", [1, 2, 3], 10)
    assert ch.take("r1") is not None
    assert ch.take("r1") is None


def test_no_offer_means_the_patch_falls_through_to_stock_behaviour():
    assert ch.take("never-offered") is None
    assert ch.active() == {}


def test_release_is_idempotent_and_safe_without_an_offer():
    ch.release("nothing")
    ch.offer("r2", [0], 0)
    ch.release("r2")
    ch.release("r2")
    assert ch.stats() == {"offered": 0, "active": 0}


def test_positions_survive_as_int64_for_the_runner_slice():
    """The runner assigns these straight into positions_np, which is int64."""
    ch.offer("r3", [0, 1, 2], 4)
    positions, _ = ch.take("r3")
    assert positions.dtype == np.int64


def test_the_runner_slice_arithmetic_reproduces_the_gap_list():
    """Walk the chunks the way the patched runner does and rebuild the full gap list."""
    p = plan_gaps([{"dst_start": 64, "dst_end": 256, "delta": 0}], 16, 600)
    positions = np.asarray(p.compute, dtype=np.int64)
    matched = p.filled_tokens

    rebuilt: list[int] = []
    num_computed = matched  # nothing computed yet
    for chunk in (100, 100, len(p.compute) - 200):
        c = num_computed - matched
        rebuilt.extend(positions[c : c + chunk].tolist())
        num_computed += chunk
    assert rebuilt == list(p.compute)
    assert num_computed == p.n_prompt
