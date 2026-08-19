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
    assert p.to_kv_transfer_params() == []  # nothing to copy past the prefix


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
    assert p.to_kv_transfer_params() == [
        {"dst_start": p.s_start, "dst_end": p.s_end, "delta": p.delta}
    ]


def test_edit_in_governing_message_is_repair():
    old = _msgs(5, governing_first=True)
    new = [dict(m) for m in old]
    new[0]["content"] = "always answer in German. " + new[0]["content"]
    p = plan(_state(old), _state(new), tokenize, repair_first=16)
    assert p.policy == "repair"
    assert p.repair_first == 16
    assert p.to_kv_transfer_params()[0]["dst_start"] == p.s_start + 16


def test_system_role_is_governing_by_default():
    old = [{"role": "system", "content": "you are a helpful assistant, be terse"}, *_msgs(3)]
    new = [dict(m) for m in old]
    new[0]["content"] = "you are a helpful assistant, be verbose and formal"
    assert plan(_state(old), _state(new), tokenize).policy == "repair"


def test_k_disjoint_edits_become_k_segments():
    """The old ceiling: two disjoint edits used to force a full recompute.

    They no longer do. Each surviving run of history is its own segment with its own
    delta, and only the rewritten entries are recomputed — which is what the vLLM
    connector needs, one request per segment.
    """
    old = _msgs(9)
    new = [dict(m) for m in old]
    for i in (1, 3, 5, 7):
        new[i]["content"] = "[EDITED] " + new[i]["content"]
    p = plan(_state(old), _state(new), tokenize)
    assert p.policy == "reuse"
    loads = p.to_kv_transfer_params()
    assert len(loads) == 4  # the runs after each of the four edits
    assert [ld["delta"] for ld in loads] == [9, 18, 27, 36]  # len("[EDITED] ") each time
    # segments are disjoint, ordered, and cover everything that was not rewritten
    ends = [0, *[ld["dst_end"] for ld in loads]]
    assert all(a <= b for a, b in zip(ends, ends[1:], strict=False))
    reused = p.segments[0].length + sum(ld["dst_end"] - ld["dst_start"] for ld in loads)
    assert reused > 0.5 * len(_state(new))  # 4 of 9 entries were rewritten


def test_a_moved_message_gets_a_negative_delta():
    """Swapping two messages is the case a prefix cache — and an LCS diff — cannot see."""
    old = _msgs(6)
    old[4]["content"] += " and a good deal more text, so the two deltas do not cancel"
    new = [dict(m) for m in old]
    new[1], new[4] = dict(old[4]), dict(old[1])  # a straight swap: same bytes, new places
    p = plan(_state(old), _state(new), tokenize)
    assert p.policy == "reuse"
    deltas = [seg.delta for seg in p.segments]
    assert any(d < 0 for d in deltas), deltas
    for seg in p.segments:
        assert _state(old)[seg.src_start : seg.src_end] == _state(new)[seg.dst_start : seg.dst_end]


def test_truncated_history_keeps_the_prefix_and_a_replaced_history_is_full():
    old = _msgs(5)
    p = plan(_state(old), _state(_msgs(5)[:1]), tokenize)
    # a truncation still leaves a valid reusable prefix; there is just nothing for the
    # connector to do, because vLLM's own prefix cache already holds it.
    assert p.policy == "reuse"
    assert p.to_kv_transfer_params() == []
    assert plan(
        _state(old), _state([{"role": "user", "content": "unrelated"}]), tokenize
    ).policy == ("full")


def test_governing_rule_survives_multi_span():
    """One governing edit among many still forces the repair policy for all segments."""
    old = _msgs(6, governing_first=True)
    new = [dict(m) for m in old]
    new[0]["content"] = "always answer in German. " + new[0]["content"]
    new[3]["content"] = "[EDITED] " + new[3]["content"]
    p = plan(_state(old), _state(new), tokenize, repair_first=8)
    assert p.policy == "repair"
    loads = p.to_kv_transfer_params()
    assert len(loads) == 2
    moved = [seg for seg in p.segments if seg.dst_start > 0]
    assert all(ld["dst_start"] == seg.dst_start + 8 for ld, seg in zip(loads, moved, strict=True))


def test_head_tokens_offsets_every_coordinate():
    """The prompt's system preamble is a leading segment, and shifts everything after."""
    old = _msgs(4)
    new = [dict(m) for m in old]
    new[2]["content"] = "[EDITED] " + new[2]["content"]
    a = plan(_state(old), _state(new), tokenize)
    b = plan(_state(old), _state(new), tokenize, head_tokens=100)
    assert b.total == a.total + 100
    assert (
        b.to_kv_transfer_params()[0]["dst_start"] == a.to_kv_transfer_params()[0]["dst_start"] + 100
    )


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


def test_phases_turn_k_segments_into_k_plus_one_requests():
    """``local_probe`` drives the plan as one request per segment; check the shape.

    Each request stops on the block boundary where the *next* segment begins, so the
    request after it prefix-hits exactly the connector's ``dst_start``. The last one is
    the real request: the whole prompt, with the last segment loaded.
    """
    from marathon.local_probe import _phases

    loads = [
        {"dst_start": 100, "dst_end": 500, "delta": 4},
        {"dst_start": 620, "dst_end": 900, "delta": 9},
        {"dst_start": 950, "dst_end": 1400, "delta": -20},
    ]
    phases = _phases(loads, block_size=16, n_prompt=1600)
    assert [ln for ln, _ in phases] == [112, 624, 960, 1600]
    assert phases[0][1] is None  # first request only prefills the first edited span
    assert [ld["delta"] for _, ld in phases[1:]] == [4, 9, -20]
    assert [ld["dst_start"] for _, ld in phases[1:]] == [112, 624, 960]
    assert [ld["dst_end"] for _, ld in phases[1:]] == [496, 896, 1392]  # clipped to blocks


def test_phases_drops_segments_shorter_than_a_block():
    from marathon.local_probe import _phases

    assert _phases([{"dst_start": 100, "dst_end": 110, "delta": 4}], 16, 1600) == []


def test_plan_drives_local_probe_end_to_end():
    """The probe's real path: canonical states in, one request per reused segment out."""
    from marathon.local_probe import _phases

    session = Session()
    session.turn("system", "You are a latency probe.")
    for t in range(8):
        session.turn("user", f"Turn {t}. filler content that is long enough to matter here")
        session.turn("assistant", "ok")
    old_state = session.replay()

    for i in (3, 9):  # two user messages rewritten in one turn
        session.edit(i, "[EDITED] " + session.messages[i]["content"])
    new_state = session.turn("user", "Turn 8. another message")

    p = plan(old_state, new_state, tokenize, head_tokens=8)
    assert p.policy == "reuse"  # the system entry is governing but was not touched
    loads = p.to_kv_transfer_params()
    assert len(loads) == 2 and all(ld["delta"] == 9 * (i + 1) for i, ld in enumerate(loads))
    phases = _phases(loads, block_size=16, n_prompt=p.total + 40)
    assert len(phases) == 3  # k=2 edits -> k+1 requests
    assert phases[-1][0] == p.total + 40  # the last one is the real, full request
