"""CPU tests for the payload -> verified state -> reuse plan pipeline.

No vLLM and no transformers: a fake engine records what it was asked to generate and a
fake tokenizer stands in for the chat template, so the part under test is the part that
has to be right on every turn — that a reconstruction is proven before it is used, that
one session's plan can never be computed against another's state, and that an ordinary
append-only turn asks the KV layer for nothing at all.
"""

from __future__ import annotations

import base64

import pytest

from marathon import client as mclient
from marathon.protocol import IntegrityError, UnknownBaselineError
from marathon.server import MarathonServer


class FakeTokenizer:
    """Byte-per-token stand-in for a chat template: one piece per message."""

    def prompt(self, messages):
        pieces = [list(f"{m['role']}: {m['content']}\n".encode()) for m in messages]
        return [i for p in pieces for i in p] + [0], pieces


class FakeEngine:
    """Records every generate call so tests can assert on the requests, not the text."""

    block_size = 16

    def __init__(self):
        self.calls = []

    def generate(self, ids, session, max_tokens, load=None, save=False):
        self.calls.append(
            {
                "n": len(ids),
                "session": session,
                "max_tokens": max_tokens,
                "load": load,
                "save": save,
            }
        )
        return f"ok {len(self.calls)}"


def make_server(**kw):
    engine = FakeEngine()
    return MarathonServer(engine=engine, tokenizer=FakeTokenizer(), **kw), engine


def make_client(server):
    return mclient.Client(mclient.local(server))


LONG = "filler " * 200  # several blocks of tokens, so segments survive block clipping


# --- hash checking ---------------------------------------------------------------


def test_tampered_delta_is_rejected():
    """A payload whose delta does not reproduce target_hash must not be trusted."""
    server, engine = make_server()
    c = make_client(server)
    c.turn("s", "hello " + LONG)

    session = c.session("s")
    session.turn("user", "second " + LONG)
    payload = session.last_payload.to_dict()
    # flip a byte inside the inserted literal, keeping target_hash as declared
    ops = payload["delta"]["ops"]
    i, op = next((i, o) for i, o in enumerate(ops) if o[0] == "i")
    raw = bytearray(base64.b64decode(op[1]))
    raw[0] ^= 0x20
    ops[i] = ["i", base64.b64encode(bytes(raw)).decode()]

    before = len(engine.calls)
    with pytest.raises(IntegrityError):
        server.turn("s", payload)
    assert len(engine.calls) == before, "a rejected payload must never reach the engine"


def test_unknown_baseline_is_rejected():
    server, engine = make_server()
    c = make_client(server)
    c.turn("s", "hello " + LONG)
    payload = c.session("s").last_payload.to_dict()
    payload["baseline_hash"] = "sha256:" + "00" * 32
    with pytest.raises(UnknownBaselineError):
        server.turn("s", payload)


def test_target_hash_mismatch_is_rejected():
    server, _ = make_server()
    c = make_client(server)
    c.turn("s", "hello " + LONG)
    c.session("s").turn("user", "second " + LONG)
    payload = c.session("s").last_payload.to_dict()
    payload["target_hash"] = "sha256:" + "11" * 32
    with pytest.raises(IntegrityError):
        server.turn("s", payload)


# --- append-only turns -----------------------------------------------------------


def test_append_only_turn_is_pure_reuse():
    """Nothing changed before the new message, so the KV layer is asked for nothing."""
    server, engine = make_server()
    c = make_client(server)
    c.turn("s", "hello " + LONG)
    r = c.turn("s", "again " + LONG)

    assert r["policy"] == "reuse"
    assert r["reason"] == "append-only"
    assert r["segments"] == 1  # one leading run: the whole previous history
    assert r["reused_tokens"] == 0  # the leading prefix is vLLM's own cache, not ours
    assert r["phases"] == 1  # one ordinary request
    assert engine.calls[-1]["load"] is None
    assert engine.calls[-1]["save"] is True  # append-only turns fill the store


def test_append_only_wire_bytes_track_the_new_text_not_the_history():
    server, _ = make_server()
    c = make_client(server)
    c.turn("s", "hello " + LONG)
    for _ in range(6):
        r = c.turn("s", "again " + LONG)
    assert r["wire_bytes"] < r["state_bytes"] / 2


# --- edits ------------------------------------------------------------------------


def test_edit_turn_plans_reuse_and_hands_segments_to_the_connector():
    server, engine = make_server()
    c = make_client(server)
    for i in range(6):
        c.turn("s", f"turn {i} " + LONG)
    c.edit("s", 0, "REWRITTEN " + LONG)
    r = c.turn("s", "after the edit " + LONG)

    assert r["segments"] == 1, "history after a leading edit survives as one shifted run"
    assert r["reused_tokens"] > 0
    assert r["phases"] == 2, "one segment is handed over as a warm-up plus the real request"
    assert engine.calls[-1]["load"] is not None
    assert engine.calls[-1]["save"] == "full", (
        "an edit turn moves the reused span, so the store must be rebuilt at the new "
        "positions or the next edit plans against a layout that no longer exists"
    )


def test_governing_edit_switches_the_policy_to_repair():
    server, _ = make_server()
    c = make_client(server)
    c.turn("s", "you are a helpful assistant", role="system")
    for i in range(4):
        c.turn("s", f"turn {i} " + LONG)
    c.edit("s", 0, "you are a terse assistant")
    r = c.turn("s", "after the edit " + LONG)
    assert r["policy"] == "repair"


def test_the_store_key_is_stable_across_edits():
    """The session keeps one store: an edit rebuilds it, it does not abandon it."""
    server, engine = make_server()
    c = make_client(server)
    for i in range(6):
        c.turn("s", f"turn {i} " + LONG)
    c.edit("s", 0, "REWRITTEN " + LONG)
    c.turn("s", "after the edit " + LONG)
    c.turn("s", "and another " + LONG)
    assert {call["session"] for call in engine.calls} == {"s"}


def test_every_edit_in_a_session_gets_the_same_treatment():
    """The 2nd and 3rd edit must reuse exactly like the 1st, not degrade to recompute."""
    server, engine = make_server()
    c = make_client(server)
    edits = []
    for i in range(21):
        if i in (8, 14, 20):
            c.edit("s", 0, f"REWRITTEN {i} " + LONG)
        r = c.turn("s", f"turn {i} " + LONG)
        if i in (8, 14, 20):
            edits.append(r)

    assert len(edits) == 3
    for n, r in enumerate(edits):
        assert r["reused_tokens"] > 0, f"edit {n} reused nothing"
        assert r["phases"] == 2, f"edit {n} did not hand a segment to the connector"
        assert r["policy"] == "reuse", f"edit {n} policy was {r['policy']}"
    # each edit reuses more than the last, because the history is longer each time
    assert edits[0]["reused_tokens"] < edits[1]["reused_tokens"] < edits[2]["reused_tokens"]
    assert all(call["save"] in (True, "full") for call in engine.calls if call["max_tokens"] > 1)


def test_full_save_after_an_edit_restores_a_contiguous_store():
    """The bookkeeping the fix rests on, checked against the real store.

    A ``"full"`` save is a save at position 0, which ``ShiftStore.reserve`` treats as a
    truncating rewrite: afterwards the session holds ``[base, new_len)`` contiguously
    and nothing above it, so the *next* edit's ``covers`` check succeeds over the whole
    new sequence. Without it the store still claims the pre-edit length, and a load for
    a position past that is declined.
    """
    torch = pytest.importorskip("torch")
    assert torch  # the store imports torch; no tensors are allocated here
    from marathon.shift_store import ShiftStore

    store = ShiftStore(budget_tokens=4096, allocate=False)
    assert store.reserve("s", 0, 1000)  # an append-only session, positions [0, 1000)
    assert store.covers("s", 0, 1000)

    # an edit turn: the new sequence is longer and everything after the edit moved
    assert store.covers("s", 400, 600), "the pre-edit layout is what the load reads"
    assert store.reserve("s", 0, 1200), "the full re-save rewrites from the start"
    assert store.covers("s", 0, 1200), "the store now describes the new coordinates"

    # the next edit can read anywhere in the new sequence, including past the old end
    assert store.covers("s", 1000, 200)
    # and an incremental save on the following append-only turn still lands
    assert store.reserve("s", 1200, 300)
    assert store.covers("s", 0, 1500)


def test_store_refuses_a_save_that_would_leave_a_hole():
    """The safety net behind the fix: a gap is refused, never silently filled."""
    pytest.importorskip("torch")
    from marathon.shift_store import ShiftStore

    store = ShiftStore(budget_tokens=4096, allocate=False)
    assert store.reserve("s", 0, 1000)
    assert not store.reserve("s", 1200, 100), "a save past `filled` would leave a hole"
    assert not store.covers("s", 900, 400), "and the span past the end is not claimed"


# --- session isolation ------------------------------------------------------------


def test_plans_are_isolated_per_session():
    """Interleaved sessions: an edit in one must not appear in the other's plan."""
    server, engine = make_server()
    c = make_client(server)
    for i in range(5):
        for sid in ("a", "b"):
            c.turn(sid, f"{sid} turn {i} " + LONG)

    c.edit("a", 0, "REWRITTEN " + LONG)
    ra = c.turn("a", "a after edit " + LONG)
    rb = c.turn("b", "b after no edit " + LONG)

    assert ra["policy"] == "reuse" and ra["segments"] == 1 and ra["reused_tokens"] > 0
    assert rb["reason"] == "append-only", "session b never changed and must stay append-only"
    assert rb["reused_tokens"] == 0
    assert engine.calls[-1]["load"] is None


def test_connector_session_keys_never_collide():
    server, engine = make_server()
    c = make_client(server)
    for sid in ("a", "b"):
        c.turn(sid, f"{sid} hello " + LONG)
    keys = {call["session"].split("#")[0] for call in engine.calls}
    assert keys == {"a", "b"}


def test_sessions_do_not_leak_content():
    """Session b's prompt must contain only session b's history."""
    server, engine = make_server()
    c = make_client(server)
    c.turn("a", "a " + LONG * 3)
    c.turn("b", "b hello")
    assert engine.calls[-1]["n"] < engine.calls[0]["n"] / 3


# --- HTTP path --------------------------------------------------------------------


def test_http_endpoint_round_trip():
    import threading

    from marathon.server import serve

    server, _ = make_server()
    http = serve(server, "127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        port = http.server_address[1]
        c = mclient.Client(mclient.http(f"http://127.0.0.1:{port}"))
        c.turn("s", "hello " + LONG)
        r = c.turn("s", "again " + LONG)
        assert r["reason"] == "append-only"
        assert r["reply"].startswith("ok")
    finally:
        http.shutdown()
        http.server_close()


def test_http_rejects_a_tampered_payload_without_dying():
    import json
    import threading
    import urllib.error
    import urllib.request

    from marathon.server import serve

    server, _ = make_server()
    http = serve(server, "127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        port = http.server_address[1]
        c = mclient.Client(mclient.local(server))
        c.turn("s", "hello " + LONG)
        c.session("s").turn("user", "second " + LONG)
        payload = c.session("s").last_payload.to_dict()
        payload["target_hash"] = "sha256:" + "22" * 32
        body = json.dumps({"session": "s", "payload": payload}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/turn",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 409
        assert json.loads(e.value.read())["error"] == "IntegrityError"
    finally:
        http.shutdown()
        http.server_close()


# --- cold tier (Phase 2) ---------------------------------------------------------


class CountingTokenizer(FakeTokenizer):
    """FakeTokenizer plus the ``encode`` the paging budget counts messages with."""

    def encode(self, text):
        return list(text.encode())


def cold_server(**kw):
    engine = FakeEngine()
    return MarathonServer(engine=engine, tokenizer=CountingTokenizer(), **kw), engine


def test_active_window_bounds_the_prompt_on_a_long_session():
    """The exit criterion, end to end: prompt tokens stop growing with the session."""
    server, engine = cold_server(active_window=1500)
    c = make_client(server)
    sizes = []
    for i in range(40):
        sizes.append(c.turn("s", f"Entry {i}. " + LONG)["active_tokens"])
    assert max(sizes[20:]) <= max(sizes[:10]) * 1.3
    assert server.cold_for("s").demoted
    # ... and without the window it grows without bound, which is the thing being fixed
    plain, _ = cold_server()
    pc = make_client(plain)
    grows = [pc.turn("p", f"Entry {i}. " + LONG)["active_tokens"] for i in range(40)]
    assert grows[-1] > 5 * grows[0]


def test_metrics_report_the_cold_tier_per_turn():
    server, _ = cold_server(active_window=1500)
    c = make_client(server)
    out = [c.turn("s", f"Entry {i}. " + LONG) for i in range(20)]
    assert all({"active_tokens", "cold_count", "promotions", "demotions"} <= set(o) for o in out)
    assert out[-1]["cold_count"] > 0
    assert any(o["demotions"] for o in out)
    assert all("reason" in d for o in out for d in o["demotions"])


def test_demotion_flows_through_the_reuse_plan_as_an_ordinary_edit():
    """A demotion must be a shrink edit the plan can reuse around, not a cache wipe."""
    # a window many messages wide, so most paging turns demote without also having to
    # evict a stub: a demotion keeps every later message at the same index, which is a
    # pure shift the plan can reuse around. (An eviction *deletes* a line, which
    # relocates everything after it, and the plan recomputes relocated blocks by design.)
    server, _ = cold_server(active_window=12000)
    c = make_client(server)
    rows = [c.turn("s", f"Entry {i}. " + LONG) for i in range(30)]
    demote_only = [
        r
        for r in rows
        if r["demotions"] and not any("evicted" in d["reason"] for d in r["demotions"])
    ]
    assert demote_only
    assert all(r["policy"] == "reuse" for r in demote_only), [r["reason"] for r in demote_only]
    # Paging makes *every* turn an edit turn, so the staleness ceiling (max_stale, see
    # findings 2026-08-19) makes consecutive demotions alternate between reusing and
    # spending one honest recompute. A demotion is still a shrink edit the plan reuses
    # around -- which is what this test is about -- on every turn the ceiling allows.
    assert all((r["reused_tokens"] > 0) != r["refreshed"] for r in demote_only)
    assert any(r["reused_tokens"] > 0 for r in demote_only)


def test_the_governing_system_prompt_is_never_paged_out():
    server, _ = cold_server(active_window=800)
    c = make_client(server)
    c.turn("s", "Standing instruction: always answer in one sentence.", role="system")
    for i in range(30):
        c.turn("s", f"Entry {i}. " + LONG)
    tier = server.cold_for("s")
    assert tier.demoted and 0 not in tier.demoted
    assert server._full["s"][0]["content"].startswith("Standing instruction")


def test_an_edit_inside_a_demoted_message_promotes_it_back():
    server, _ = cold_server(active_window=1500)
    c = make_client(server)
    for i in range(25):
        c.turn("s", f"Entry {i}. " + LONG)
    tier = server.cold_for("s")
    target = sorted(tier.demoted)[0]
    c.edit("s", target, f"CORRECTED entry. The beacon code is 4821-OMEGA. {LONG}")
    out = c.turn("s", "carry on")
    assert target in [p["index"] for p in out["promotions"]]
    assert target not in tier.demoted


def test_paging_is_isolated_per_session():
    server, _ = cold_server(active_window=1200)
    c = make_client(server)
    for i in range(25):
        c.turn("a", f"Entry {i}. " + LONG)
    c.turn("b", "short first turn")
    assert server.cold_for("a").demoted
    assert not server.cold_for("b").demoted
    assert server.cold_for("a") is not server.cold_for("b")


def test_no_active_window_leaves_the_pipeline_untouched():
    server, _ = cold_server()
    out = make_client(server).turn("s", "hello " + LONG)
    assert server.cold_for("s") is None
    assert out["cold_count"] == 0 and out["promotions"] == [] and out["demotions"] == []


# --- staleness: replay turns, and the churn ceiling ------------------------------


def test_churn_tokens_measures_what_changed_in_front_of_the_span():
    from marathon.shift_store import churn_tokens

    assert churn_tokens([]) == (0, 0)
    # one span starting at 500, nothing reused before it: 500 tokens changed in front
    assert churn_tokens([{"dst_start": 500, "dst_end": 1500, "delta": -8}]) == (500, 1000)
    # an unchanged prefix reused as its own segment is *not* churn
    loads = [
        {"dst_start": 0, "dst_end": 400, "delta": 0},
        {"dst_start": 500, "dst_end": 1500, "delta": -8},
    ]
    assert churn_tokens(loads) == (100, 1400)
    # append-only: a single segment from 0 changes nothing in front of itself
    assert churn_tokens([{"dst_start": 0, "dst_end": 900, "delta": 0}]) == (0, 900)


def test_a_replay_turn_does_not_advance_the_staleness_clock():
    """``generate=False`` saves no KV and reuses none, so it must not ratchet.

    Regression test for the interaction Track L flagged: replaying a history to bring a
    session's paging state up to date would otherwise force spurious refreshes on the
    real turns that follow it.
    """
    server, _ = cold_server(active_window=1500, max_stale=1)
    session = mclient.Client(mclient.local(server)).session("s")
    for i in range(12):
        session.turn("user", f"Entry {i}. " + LONG)
        server.turn("s", session.last_payload.to_dict(), generate=False)
        session.messages.append({"role": "assistant", "content": "ok"})
    assert server._stale.get("s", 0) == 0
    assert server._churn.get("s", 0) == 0


def test_replay_then_generate_leaves_the_clock_consistent():
    """A replayed history must not change what the first *real* turn is allowed to do."""
    results = {}
    for replay in (False, True):
        server, _ = cold_server(active_window=1500, max_stale=1)
        c = make_client(server)
        session = c.session("s")
        for i in range(10):
            session.turn("user", f"Entry {i}. " + LONG)
            server.turn("s", session.last_payload.to_dict(), generate=not replay)
            session.messages.append({"role": "assistant", "content": "ok"})
        session.turn("user", "final question")
        out = server.turn("s", session.last_payload.to_dict())
        results[replay] = (out["stale"], out["refreshed"])
    assert results[True] == results[False], results


def test_max_stale_forces_a_refresh_on_consecutive_reused_edit_turns():
    server, _ = cold_server(active_window=12000, max_stale=1)
    c = make_client(server)
    rows = [c.turn("s", f"Entry {i}. " + LONG) for i in range(30)]
    reused = [r for r in rows if r["reused_tokens"]]
    assert reused, "no turn reused anything; the test is not exercising the ceiling"
    assert any(r["refreshed"] for r in rows)
    # a refresh turn reuses nothing, by definition
    assert all(r["reused_tokens"] == 0 for r in rows if r["refreshed"])
    # and the clock never exceeds the ceiling
    assert max(r["stale"] for r in rows) <= 1


def test_max_churn_replaces_the_turn_counter():
    """With ``max_churn`` set, the turn counter is not what decides a refresh."""
    server, _ = cold_server(active_window=12000, max_stale=1, max_churn=0.2)
    c = make_client(server)
    rows = [c.turn("s", f"Entry {i}. " + LONG) for i in range(30)]
    assert any(r["reused_tokens"] for r in rows)
    assert all(r["churn"] >= 0.0 for r in rows)
    # the ceiling holds: a turn is only allowed to reuse while accumulated churn is under
    for r in rows:
        if r["refreshed"]:
            assert r["churn"] > 0.2
    # and it is genuinely a different policy from counting turns
    strict, _ = cold_server(active_window=12000, max_stale=1)
    c2 = make_client(strict)
    turn_rows = [c2.turn("t", f"Entry {i}. " + LONG) for i in range(30)]
    assert [r["refreshed"] for r in rows] != [r["refreshed"] for r in turn_rows]


def test_churn_accumulates_across_consecutive_reused_turns_and_resets_on_refresh():
    """Small turns churn the span slowly, so the ceiling takes several of them to trip."""
    short = "filler " * 40
    server, _ = cold_server(active_window=6000, max_stale=99, max_churn=0.2)
    c = make_client(server)
    rows = [c.turn("s", f"Entry {i}. " + short) for i in range(60)]
    refreshed_at = [i for i, r in enumerate(rows) if r["refreshed"]]
    assert refreshed_at, "churn never crossed the threshold; nothing was exercised"
    # a refresh empties the accumulator, so the next turn starts from its own churn only
    for i in refreshed_at:
        assert rows[i]["churn"] > 0.2
    # within a run of consecutive reused turns the accumulator only grows
    run: list[float] = []
    for r in rows:
        if r["refreshed"] or not r["reused_tokens"]:
            assert run == sorted(run), run
            run = []
        else:
            run.append(r["churn"])
    assert run == sorted(run), run
    # the turn counter is deliberately not the thing being enforced here
    assert max(r["stale"] for r in rows) > 1
