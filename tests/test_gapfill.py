"""The plan -> non-prefix match mask logic the single-request load path rests on.

No vLLM here: this is the arithmetic the patched scheduler and model runner consume, and
it has to be right before any of it runs on a GPU. The properties that matter are that
the filled spans and the computed positions exactly partition the prompt, that the engine
is always left the final position (or there is no forward pass to run), and that spans
too small to survive block alignment are dropped rather than silently rounded outward
onto positions the connector will not actually fill.
"""

from __future__ import annotations

import pytest

from marathon.gapfill import GapPlan, align, check, plan_gaps

BLOCK = 16


def load(lo: int, hi: int, delta: int = 0) -> dict:
    return {"dst_start": lo, "dst_end": hi, "delta": delta}


def test_no_loads_means_compute_everything():
    p = plan_gaps([], BLOCK, 500)
    assert p.filled == ()
    assert list(p.compute) == list(range(500))
    assert p.saving == 0.0


def test_single_segment_partitions_the_prompt():
    p = plan_gaps([load(100, 400)], BLOCK, 1000)
    assert p.filled == ((112, 400),)
    assert set(p.compute) == set(range(112)) | set(range(400, 1000))
    assert len(p.compute) + p.filled_tokens == 1000


def test_multi_segment_gaps_are_exactly_the_complement():
    p = plan_gaps([load(100, 400), load(600, 900)], BLOCK, 1000)
    assert p.filled == ((112, 400), (608, 896))
    covered = {i for lo, hi in p.filled for i in range(lo, hi)}
    assert covered.isdisjoint(p.compute)
    assert covered | set(p.compute) == set(range(1000))


def test_a_span_smaller_than_a_block_is_dropped_not_rounded_out():
    """Rounding outward would claim positions the connector never fills."""
    p = plan_gaps([load(100, 110)], BLOCK, 500)
    assert p.filled == ()
    assert len(p.compute) == 500


def test_the_last_block_is_never_claimed():
    """The engine must be left something to compute, whatever the plan asks for."""
    p = plan_gaps([load(0, 1000)], BLOCK, 1000)
    assert p.filled[-1][1] < 1000
    assert p.compute[-1] == 999


def test_touching_spans_are_merged():
    p = plan_gaps([load(0, 256), load(256, 512)], BLOCK, 1000)
    assert p.filled == ((0, 512),)


def test_overlapping_spans_are_merged_not_double_counted():
    p = plan_gaps([load(0, 300), load(200, 512)], BLOCK, 1000)
    assert p.filled == ((0, 512),)
    assert p.filled_tokens == 512
    assert len(p.compute) == 1000 - 512


def test_unsorted_loads_are_handled():
    a = plan_gaps([load(600, 900), load(100, 400)], BLOCK, 1000)
    b = plan_gaps([load(100, 400), load(600, 900)], BLOCK, 1000)
    assert a == b


@pytest.mark.parametrize("n_prompt", [17, 64, 513, 1000, 7939])
@pytest.mark.parametrize("block", [16, 32])
def test_partition_property_holds_across_shapes(n_prompt: int, block: int):
    loads = [load(n_prompt // 8, n_prompt // 3), load(n_prompt // 2, n_prompt - block)]
    p = plan_gaps(loads, block, n_prompt)
    covered = {i for lo, hi in p.filled for i in range(lo, hi)}
    assert covered | set(p.compute) == set(range(n_prompt))
    assert covered.isdisjoint(p.compute)
    assert p.compute[-1] == n_prompt - 1


def test_check_rejects_a_mask_that_would_starve_the_forward_pass():
    """`check` is the contract the patch trusts; it must actually reject bad masks."""
    with pytest.raises(AssertionError):
        check(GapPlan(filled=((0, 16),), compute=(), n_prompt=16))
    with pytest.raises(AssertionError):
        check(GapPlan(filled=((0, 16), (8, 32)), compute=(32,), n_prompt=33))
    with pytest.raises(AssertionError):
        check(GapPlan(filled=((0, 16),), compute=(16, 16, 17), n_prompt=33))


def test_align_matches_the_phase_drivers_clipping_rule():
    """Same block-alignment as reuse_plan.phases, so the two paths reuse the same spans."""
    from marathon.reuse_plan import phases

    loads = [load(100, 400, -5), load(600, 900, -9)]
    aligned = align(loads, BLOCK, 1000)
    ph = phases(loads, BLOCK, 1000)
    assert [(lo, hi) for lo, hi in aligned] == [(d["dst_start"], d["dst_end"]) for _, d in ph if d]


def test_the_engines_own_prefix_hit_counts_as_filled():
    """Positions the engine already has must not be handed back to it to recompute."""
    p = plan_gaps([load(400, 800)], BLOCK, 1000, local_hit=112)
    assert p.filled[0] == (0, 112)
    assert min(p.compute) == 112
    covered = {i for lo, hi in p.filled for i in range(lo, hi)}
    assert covered | set(p.compute) == set(range(1000))
    assert covered.isdisjoint(p.compute)


def test_local_hit_merges_with_a_segment_that_starts_inside_it():
    p = plan_gaps([load(64, 400)], BLOCK, 1000, local_hit=128)
    assert p.filled == ((0, 400),)
    assert p.compute[0] == 400


def test_local_hit_alone_still_leaves_the_last_token():
    p = plan_gaps([], BLOCK, 500, local_hit=500)
    assert p.compute[-1] == 499
    assert p.filled_tokens + len(p.compute) == 500
