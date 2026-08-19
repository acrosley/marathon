"""Policy tests for the reuse plan: which edits are safe to stitch, and which are not.

Tokenization is stubbed as bytes -> ids 1:1, so token offsets are byte offsets and the
spans can be asserted exactly without pulling a real tokenizer into CI.
"""

from __future__ import annotations

import pytest

from marathon.canonical import serialize_history
from marathon.session import Session

reuse_plan = pytest.importorskip("marathon.reuse_plan")  # imports torch via kvshift
plan = reuse_plan.plan


def tokenize(data: bytes) -> list[int]:
    return list(data)


def _state(messages: list[dict]) -> bytes:
    return serialize_history(messages)


def _msgs(n: int, governing_first: bool = False) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        m = {"role": "user", "content": f"message number {i} with some filler content"}
        if i == 0 and governing_first:
            m["governing"] = True
        out.append(m)
    return out


def test_append_only_is_reuse():
    old = _msgs(4)
    p = plan(_state(old), _state([*old, {"role": "user", "content": "new turn"}]), tokenize)
    assert p.policy == "reuse"
    assert p.p == len(_state(old))
    assert p.delta == 0
    assert p.e_start == p.p and p.e_end > p.e_start  # E' is the appended tail
    assert p.to_kv_transfer_params() is None  # nothing to copy: S is empty


def test_no_change_is_reuse():
    old = _msgs(3)
    p = plan(_state(old), _state(old), tokenize)
    assert p.policy == "reuse" and p.delta == 0


def test_edit_in_non_governing_message_is_reuse():
    old = _msgs(5)
    new = [dict(m) for m in old]
    new[1]["content"] = "[EDITED] " + new[1]["content"]
    p = plan(_state(old), _state(new), tokenize)
    assert p.policy == "reuse"
    assert p.repair_first == 0
    assert p.delta == len("[EDITED] ")
    assert p.s_end == len(_state(new))
    assert p.to_kv_transfer_params() == {
        "dst_start": p.s_start,
        "dst_end": p.s_end,
        "delta": p.delta,
    }


def test_edit_in_governing_message_is_repair():
    old = _msgs(5, governing_first=True)
    new = [dict(m) for m in old]
    new[0]["content"] = "always answer in German. " + new[0]["content"]
    p = plan(_state(old), _state(new), tokenize, repair_first=16)
    assert p.policy == "repair"
    assert p.repair_first == 16
    assert p.to_kv_transfer_params()["dst_start"] == p.s_start + 16


def test_system_role_is_governing_by_default():
    old = [{"role": "system", "content": "you are a helpful assistant, be terse"}, *_msgs(3)]
    new = [dict(m) for m in old]
    new[0]["content"] = "you are a helpful assistant, be verbose and formal"
    assert plan(_state(old), _state(new), tokenize).policy == "repair"


def test_two_edits_are_full():
    old = _msgs(5)
    new = [dict(m) for m in old]
    new[1]["content"] = "[EDITED] " + new[1]["content"]
    new[3]["content"] = "[EDITED] " + new[3]["content"]
    p = plan(_state(old), _state(new), tokenize)
    assert p.policy == "full"
    assert p.to_kv_transfer_params() is None


def test_session_turn_marks_governing_and_keeps_bytes_stable():
    plain, marked = Session(), Session()
    plain.turn("user", "hello")
    marked.turn("user", "hello", governing=True)
    assert plain.replay() != marked.replay()  # opt-in flag is serialized
    assert b"governing" not in plain.replay()  # ...and only when set
    sys_session = Session()
    sys_session.turn("system", "be terse")
    assert sys_session.messages[0]["governing"] is True
    sys_session.edit(0, "be verbose")
    assert sys_session.messages[0]["governing"] is True  # edit preserves the flag


def test_matches_local_probe_reuse_plan():
    """``to_kv_transfer_params`` reproduces what ``local_probe`` computes today.

    ``local_probe._reuse_plan`` works on the probe's chunked prompt; feeding it the
    canonical JSONL lines as chunks (and block_size 1, so it does no block rounding)
    puts both on the same coordinates.
    """
    from marathon.local_probe import _reuse_plan

    session = Session()
    session.turn("system", "You are a latency probe.")  # a=0 in _reuse_plan: never edited
    for t in range(4):
        session.turn("user", f"Turn {t}. filler content that is long enough to matter")
        session.turn("assistant", "ok")
    old_state = session.replay()

    session.edit(1, "[EDITED] " + session.messages[1]["content"])
    new_state = session.turn("user", "Turn 4. another message")

    old_chunks = [list(line) for line in reuse_plan._lines(old_state)]
    new_chunks = [list(line) for line in reuse_plan._lines(new_state)]
    phase1_len, dst_end, delta = _reuse_plan(old_chunks, new_chunks, block_size=1)

    got = plan(old_state, new_state, tokenize).to_kv_transfer_params()
    assert got["delta"] == delta
    assert got["dst_end"] == dst_end
    # local_probe is chunk-granular: S starts at the next whole message. token_span
    # snaps to the actual divergence, so S starts earlier — inside the edited message,
    # right after the inserted bytes. Same span, tighter start; strictly more reuse.
    assert got["dst_start"] <= phase1_len
