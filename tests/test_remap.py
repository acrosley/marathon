"""The generation-0 address book: reuse without ever stitching from stitched bytes.

The property under test is the one the whole no-resave idea rests on: after any number of
reuse turns, the delta a load carries equals the distance from the span's *generation-0*
store index to where the text sits now. If that holds, one exact rotation reproduces the
original KV however many times the text has moved, and reused KV never compounds.
"""

from __future__ import annotations

from marathon.remap import Remap, loads_for


def seg(lo: int, hi: int, delta: int) -> dict:
    return {"dst_start": lo, "dst_end": hi, "delta": delta}


def test_empty_remap_is_the_identity():
    r = Remap()
    assert r.translate(0, 100) == [(0, 100, 0)]
    assert loads_for(r, [seg(100, 200, -50)]) == [{"dst_start": 100, "dst_end": 200, "delta": -50}]


def test_one_reuse_turn_records_the_delta():
    r = Remap().after_turn([(100, 400, -50)], total=1000)
    assert r.pieces == [(100, 400, -50)]
    # the span now at 100..400 is backed by store 150..450
    assert r.translate(100, 400) == [(100, 400, -50)]


def test_offsets_compose_additively_across_turns():
    """Two shifts of -50 must leave the span reachable by ONE rotation of -100.

    Note the order: the loads for a turn are read off the map as it stood *before* it,
    because the plan's delta is measured against the previous state.
    """
    r0 = Remap().after_turn([(100, 400, -50)], total=1000)
    loads = loads_for(r0, [seg(50, 350, -50)])
    assert loads == [{"dst_start": 50, "dst_end": 350, "delta": -100}]
    r1 = r0.after_turn([(50, 350, -50)], total=1000)
    assert r1.pieces == [(50, 350, -100)]


def test_depth_grows_with_reuse_and_resets_where_text_is_recomputed():
    r = Remap()
    for _ in range(5):
        r = r.after_turn([(100, 400, -20)], total=1000)
    assert r.depth() == 100
    # a turn that reuses nothing leaves nothing mapped: everything was recomputed
    assert Remap().after_turn([], total=1000).pieces == []


def test_a_span_crossing_two_offsets_splits_into_two_loads():
    """Different parts of the history drift by different amounts and need own rotations."""
    r = Remap(pieces=[(0, 100, -10), (100, 200, -30)])
    loads = loads_for(r, [seg(0, 200, 0)])
    assert loads == [
        {"dst_start": 0, "dst_end": 100, "delta": -10},
        {"dst_start": 100, "dst_end": 200, "delta": -30},
    ]


def test_uncovered_regions_are_offset_zero():
    """Freshly computed text is stored where it belongs and needs no extra rotation."""
    r = Remap(pieces=[(100, 200, -30)])
    assert r.translate(0, 300) == [(0, 100, 0), (100, 200, -30), (200, 300, 0)]


def test_recomputed_text_drops_out_of_the_map():
    """A turn only maps what it reused; anything else was saved at its own index."""
    r = Remap(pieces=[(0, 400, -50)])
    r = r.after_turn([(0, 100, 0)], total=1000)
    assert r.pieces == [(0, 100, -50)]
    assert r.translate(100, 400) == [(100, 400, 0)]


def test_translate_partitions_exactly():
    r = Remap(pieces=[(50, 150, -5), (300, 400, -9)])
    pieces = r.translate(0, 500)
    assert pieces[0][0] == 0 and pieces[-1][1] == 500
    for (_, a_hi, _), (b_lo, _, _) in zip(pieces, pieces[1:], strict=False):
        assert a_hi == b_lo


def test_generation_zero_invariant_over_a_long_paged_session():
    """The load's delta must always be (where it sits now) - (where it was computed).

    Simulated as the pager does it: a front demotion shrinks the view by 200 tokens every
    turn, so the tail keeps sliding earlier. Whatever turn we are on, one rotation by the
    recorded offset has to reach the original index.
    """
    total, span_lo, span_hi = 10000, 6000, 9000
    r = Remap()
    origin = {p: p for p in range(span_lo, span_hi)}  # logical -> generation-0 index
    for _ in range(12):
        delta = -200
        r = r.after_turn([(span_lo + delta, span_hi + delta, delta)], total=total)
        origin = {p + delta: origin[p] for p in list(origin)}
        span_lo, span_hi = span_lo + delta, span_hi + delta

        for load in loads_for(r, [seg(span_lo, span_hi, 0)]):
            src = load["dst_start"] - load["delta"]
            assert src == origin[load["dst_start"]], (
                f"load reaches store {src}, generation-0 index is {origin[load['dst_start']]}"
            )
    assert r.depth() == 2400


def test_merge_joins_neighbours_and_clips_to_the_prompt():
    r = Remap().after_turn([(0, 100, -5), (100, 200, -5)], total=150)
    assert r.pieces == [(0, 150, -5)]
