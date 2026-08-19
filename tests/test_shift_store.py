"""CPU tests for the shift connector's bookkeeping: sessions, budget, positions.

Everything here runs without vLLM and without a GPU — the KV layout is faked with
small CPU tensors, which is the whole reason ``shift_store`` is a separate module
from ``vllm_shift_connector``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marathon.shift_store import SessionTable, ShiftStore, slots  # noqa: E402


def _kv(n: int, value: float, heads: int = 2, dim: int = 4) -> torch.Tensor:
    return torch.full((n, heads, dim), value)


# ------------------------------------------------------------------ slot mapping


def test_slots_map_positions_through_the_block_table():
    got = slots([7, 3], 2, 6, block_size=4)
    # positions 2,3 live in block 7 at offsets 2,3; positions 4,5 in block 3.
    assert got.tolist() == [7 * 4 + 2, 7 * 4 + 3, 3 * 4 + 0, 3 * 4 + 1]


# ------------------------------------------------------------------ single writer


def test_session_table_refuses_a_second_in_flight_writer():
    t = SessionTable()
    assert t.acquire("r1", "s")
    assert not t.acquire("r2", "s")
    assert t.session_of("r2") is None
    assert t.conflicts == 1
    t.release("r1")
    assert t.acquire("r2", "s")
    assert t.session_of("r2") == "s"


def test_session_table_reacquire_by_the_same_request_is_fine():
    t = SessionTable()
    assert t.acquire("r1", "s")
    assert t.acquire("r1", "s")
    assert t.conflicts == 0


def test_session_table_release_of_an_unknown_request_is_a_noop():
    t = SessionTable()
    t.acquire("r1", "s")
    t.release("nobody")
    assert t.session_of("r1") == "s"
    assert len(t) == 1


# ------------------------------------------------------------- position accounting


def test_append_only_saves_extend_the_filled_span():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    assert st.reserve("a", 0, 100)
    assert st.reserve("a", 100, 50)
    assert st.covers("a", 0, 150)
    assert not st.covers("a", 0, 151)


def test_a_save_that_would_leave_a_hole_is_refused():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    st.reserve("a", 0, 100)
    assert not st.reserve("a", 200, 10)
    assert st.covers("a", 0, 100)
    assert st.refusals == 1


def test_an_edit_truncates_everything_above_it():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    st.reserve("a", 0, 500)
    # the edit turn recomputes from position 200 onward: what is above is stale
    assert st.reserve("a", 200, 60)
    assert st.covers("a", 0, 260)
    assert not st.covers("a", 0, 300)


def test_a_span_larger_than_the_budget_is_refused():
    st = ShiftStore(slab=1024, budget_tokens=1024, device="cpu")
    assert not st.reserve("a", 0, 2000)
    assert not st.covers("a", 0, 1)


def test_reserve_is_idempotent_so_the_worker_may_call_it_per_layer():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    for _ in range(5):
        assert st.reserve("a", 0, 300)
    assert st.stats()["sessions"] == {"a": 300}


# ------------------------------------------------------------------- KV round trip


def test_write_read_round_trip_and_session_isolation():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    st.reserve("a", 0, 8)
    st.reserve("b", 0, 8)
    st.write("a", "layer0", 0, _kv(8, 1.0))
    st.write("b", "layer0", 0, _kv(8, 2.0))
    assert torch.equal(st.read("a", "layer0", 0, 8), _kv(8, 1.0))
    assert torch.equal(st.read("b", "layer0", 0, 8), _kv(8, 2.0))
    # a session that never wrote this layer reads nothing, rather than someone else's
    assert st.read("a", "layer1", 0, 8) is None
    assert st.read("nobody", "layer0", 0, 8) is None


def test_read_past_the_filled_span_returns_none():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    st.reserve("a", 0, 8)
    st.write("a", "layer0", 0, _kv(8, 1.0))
    assert st.read("a", "layer0", 4, 8) is None


def test_growth_preserves_already_stored_kv():
    st = ShiftStore(slab=1024, budget_tokens=8192, device="cpu")
    st.reserve("a", 0, 100)
    st.write("a", "layer0", 0, _kv(100, 3.0))
    st.reserve("a", 100, 2000)  # forces a realloc past the first chunk
    st.write("a", "layer0", 100, _kv(2000, 4.0))
    assert torch.equal(st.read("a", "layer0", 0, 100), _kv(100, 3.0))
    assert torch.equal(st.read("a", "layer0", 100, 2000), _kv(2000, 4.0))


# ------------------------------------------------------------------ budget and LRU


def test_lru_evicts_the_least_recently_used_session():
    st = ShiftStore(slab=1024, budget_tokens=3072, device="cpu")
    st.reserve("a", 0, 1000)
    st.reserve("b", 0, 1000)
    st.reserve("c", 0, 1000)  # three sessions, one 1024-token slab each
    st.covers("a", 0, 1000)  # touching "a" makes "b" the least recently used
    st.reserve("c", 1000, 500)  # c must double its slab: someone has to go
    assert st.stats()["evictions"] == 1
    assert set(st.stats()["sessions"]) == {"a", "c"}
    assert not st.covers("b", 0, 10)


def test_eviction_never_takes_the_session_being_written():
    st = ShiftStore(slab=1024, budget_tokens=2048, device="cpu")
    st.reserve("a", 0, 1000)
    st.reserve("b", 0, 1000)
    st.reserve("b", 1000, 1000)  # b needs 2048; a is evicted, b survives
    assert set(st.stats()["sessions"]) == {"b"}
    assert st.covers("b", 0, 2000)


def test_an_evicted_session_is_a_miss_not_a_wrong_load():
    st = ShiftStore(slab=1024, budget_tokens=2048, device="cpu")
    st.reserve("a", 0, 1000)
    st.write("a", "layer0", 0, _kv(1000, 9.0))
    st.reserve("b", 0, 2000)
    assert not st.covers("a", 0, 1000)
    assert st.read("a", "layer0", 0, 1000) is None
    assert st.misses == 1


def test_bookkeeping_only_mode_tracks_positions_without_tensors():
    """The scheduler-side mirror: same answers to ``covers``, no memory."""
    sched = ShiftStore(slab=1024, budget_tokens=4096, device="cpu", allocate=False)
    worker = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    for start, n in ((0, 500), (500, 500), (200, 100)):
        assert sched.reserve("a", start, n) == worker.reserve("a", start, n)
    sched.write("a", "layer0", 0, _kv(8, 1.0))
    assert sched.read("a", "layer0", 0, 8) is None
    assert sched.stats()["sessions"] == worker.stats()["sessions"] == {"a": 300}


def test_stats_reports_tokens_per_session():
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    st.reserve("a", 0, 300)
    st.reserve("b", 0, 700)
    st.write("b", "layer0", 0, _kv(700, 1.0))
    s = st.stats()
    assert s["sessions"] == {"a": 300, "b": 700}
    assert s["saved_token_layers"] == 700
    assert s["budget_tokens"] == 4096


def test_the_first_save_may_start_above_zero():
    """vLLM's prefix cache serves the head, so the request never computes it."""
    st = ShiftStore(slab=1024, budget_tokens=4096, device="cpu")
    assert st.reserve("a", 16, 600)  # turn 0 prefix-hit 16 tokens of system prompt
    assert st.reserve("a", 616, 600)  # turn 1 computed only its own new tokens
    assert st.covers("a", 616, 600)
    assert st.covers("a", 16, 1200)
    # ...but the store must not claim the head it never wrote
    assert not st.covers("a", 0, 1216)
    assert st.read("a", "layer0", 0, 16) is None
