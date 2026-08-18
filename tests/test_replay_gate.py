"""Correctness gate: delta-reconstructed state == full-context replay, every turn.

Efficiency that changes answers is a regression. This test drives a session
with mid-history edits through the wire protocol and proves, at every turn,
that what the server reconstructs from deltas is byte-identical to replaying
the full logical history from scratch — and that the ledger stays verifiable.
"""

import random

from marathon.canonical import serialize_history
from marathon.session import Session


def test_reconstruction_matches_full_replay_at_every_turn(tmp_path):
    rng = random.Random(3)
    s = Session()
    log: list[dict] = []  # independent copy of the logical history
    for t in range(60):
        if t and t % 7 == 0:  # edit an earlier message: the prefix-cache-breaking case
            j = rng.randrange(len(log))
            log[j] = {**log[j], "content": log[j]["content"] + " [edited]"}
            s.edit(j, log[j]["content"])
        content = "".join(rng.choice("abcdef ,.") for _ in range(rng.randrange(1, 300)))
        log.append({"role": "user" if t % 2 == 0 else "assistant", "content": content})

        resolved = s.turn(log[-1]["role"], content)

        assert resolved == serialize_history(log)  # server bytes == full replay
        assert Session.decode(resolved) == log  # model-facing view == logical history
        assert s.last_payload.target_hash in s.store

    s.ledger.verify()
    s.ledger.to_jsonl(tmp_path / "ledger.jsonl")
    assert len(type(s.ledger).from_jsonl(tmp_path / "ledger.jsonl")) == 60
