import dataclasses

import pytest

from marathon.canonical import serialize_history
from marathon.protocol import (
    BaselineStore,
    IntegrityError,
    TurnPayload,
    UnknownBaselineError,
    prepare_turn,
    resolve_turn,
)


def test_first_turn_uses_empty_baseline():
    store = BaselineStore()
    state = serialize_history([{"turn": 0, "content": "hello"}])
    payload = prepare_turn(store, None, state, new_input="hello")
    assert payload.baseline_hash is None
    assert resolve_turn(store, payload) == state
    assert payload.target_hash in store


def test_multi_turn_session_round_trip():
    store = BaselineStore()
    history = []
    baseline = None
    for t in range(10):
        history.append({"turn": t, "content": f"message {t} " * 20})
        state = serialize_history(history)
        payload = prepare_turn(store, baseline, state, new_input=f"message {t}")
        # wire round-trip: server decodes exactly what client encoded
        decoded = TurnPayload.from_wire(payload.wire_bytes())
        assert resolve_turn(store, decoded) == state
        baseline = decoded.target_hash


def test_deltas_shrink_relative_to_full_resend():
    store = BaselineStore()
    history = []
    baseline = None
    wire_last = full_last = 0
    for t in range(20):
        history.append({"turn": t, "content": "x" * 500})
        state = serialize_history(history)
        payload = prepare_turn(store, baseline, state, new_input="x")
        resolve_turn(store, payload)
        baseline = payload.target_hash
        wire_last, full_last = len(payload.wire_bytes()), len(state)
    assert wire_last < full_last / 5


def test_unknown_baseline_rejected():
    store = BaselineStore()
    with pytest.raises(UnknownBaselineError):
        prepare_turn(store, "sha256:" + "a" * 64, b"state", new_input="x")


def test_integrity_error_on_forged_target_hash():
    store = BaselineStore()
    state = serialize_history([{"turn": 0, "content": "hello"}])
    payload = prepare_turn(store, None, state, new_input="hello")
    forged = dataclasses.replace(payload, target_hash="sha256:" + "f" * 64)
    with pytest.raises(IntegrityError):
        resolve_turn(store, forged)
