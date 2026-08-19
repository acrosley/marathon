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
    assert engine.calls[-1]["save"] is False, "an edit turn reads the store, never writes it"


def test_governing_edit_switches_the_policy_to_repair():
    server, _ = make_server()
    c = make_client(server)
    c.turn("s", "you are a helpful assistant", role="system")
    for i in range(4):
        c.turn("s", f"turn {i} " + LONG)
    c.edit("s", 0, "you are a terse assistant")
    r = c.turn("s", "after the edit " + LONG)
    assert r["policy"] == "repair"


def test_store_epoch_rolls_after_an_edit():
    """A later turn must not layer new positions over the pre-edit store layout."""
    server, engine = make_server()
    c = make_client(server)
    for i in range(6):
        c.turn("s", f"turn {i} " + LONG)
    before = engine.calls[-1]["session"]
    c.edit("s", 0, "REWRITTEN " + LONG)
    c.turn("s", "after the edit " + LONG)
    c.turn("s", "and another " + LONG)
    assert engine.calls[-1]["session"] != before


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
