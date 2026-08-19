"""CPU tests for the cold tier: paging invariants, stub determinism, recall triggers."""

from __future__ import annotations

import json

import pytest

from marathon.canonical import canonical_bytes
from marathon.cold import ColdTier, HashEmbedder, is_governing, stub_text
from marathon.session import Session


def build(n_turns: int = 30, body_words: int = 60) -> list[dict]:
    """A history long enough to page: one system prompt then ``n_turns`` user/assistant."""
    messages = [{"role": "system", "content": "You are a careful assistant.", "governing": True}]
    for t in range(n_turns):
        filler = " ".join(f"entry{t}word{w}" for w in range(body_words))
        body = f"Entry {t}. The code for {t} is {t}-ZETA. {filler}"
        messages.append({"role": "user", "content": body})
        messages.append({"role": "assistant", "content": f"Noted entry {t}."})
    return messages


def tier(**kw) -> ColdTier:
    kw.setdefault("active_tokens", 400)
    return ColdTier(**kw)


# ------------------------------------------------------------ stub determinism


def test_stub_is_deterministic_and_byte_stable():
    m = {"role": "user", "content": "The harbor code is 4821-OMEGA. " + "x " * 50}
    assert stub_text(7, m) == stub_text(7, m)
    assert stub_text(7, dict(reversed(list(m.items())))) == stub_text(7, m)  # key order
    assert stub_text(8, m) != stub_text(7, m)  # index is part of the identity


def test_stub_shape_and_hash_is_the_content_address():
    m = {"role": "user", "content": "alpha beta gamma " * 20}
    text = stub_text(3, m, words=12)
    assert text.startswith("[cold #3 ") and text.endswith("]")
    h = text.split()[2].rstrip(":")
    assert len(h) == 8
    import hashlib

    assert hashlib.sha256(canonical_bytes(m)).hexdigest().startswith(h)
    assert len(text.split(": ", 1)[1].rstrip("]").split()) == 12


def test_stub_survives_short_and_empty_messages():
    assert stub_text(0, {"role": "user", "content": ""}).endswith(": ]")
    assert "hi" in stub_text(1, {"role": "user", "content": "hi"})


def test_active_state_is_pure_in_messages_and_demoted_set():
    messages = build(10)
    a, b = tier(), tier()
    a.demoted = {3, 5, 9}
    b.demoted = {9, 5, 3}
    assert a.active_state(messages) == b.active_state(messages)
    assert a.active_state(messages) != tier().active_state(messages)


# ------------------------------------------------------- bounded-window invariant


@pytest.mark.parametrize("budget", [200, 400, 800, 1600])
def test_window_stays_bounded_on_an_unbounded_session(budget):
    """The whole point: the active window does not grow with the session."""
    t = tier(active_tokens=budget)
    messages: list[dict] = [{"role": "system", "content": "sys", "governing": True}]
    sizes = []
    for i in range(120):
        body = f"Entry {i}. " + " ".join(f"w{i}x{j}" for j in range(40))
        messages.append({"role": "user", "content": body})
        messages.append({"role": "assistant", "content": f"Noted {i}."})
        t.step(messages, None, "carry on")
        sizes.append(t.active_token_count(messages))
    # the floor is what the policy is *not allowed* to page out: governing + last K +
    # one stub per demoted message. Everything above that is bounded by the budget.
    # the floor is what the policy is *not allowed* to page out: the governing system
    # prompt plus the last K messages. Everything above that is what the budget bounds.
    biggest = max(t.count(m) for m in messages)
    floor = sum(t.count(messages[i]) for i in t.protected(messages))
    assert max(sizes) <= max(budget, floor) + biggest
    # flat, not merely sub-linear: the second half of a 120-turn session is no bigger
    # than the first (one message of slack, since the protected tail varies in size)
    assert max(sizes[60:]) <= max(sizes[20:60]) + biggest
    # and the cold tier really is doing the work rather than the session being short
    assert len(t.demoted) > 100


def test_governing_messages_are_never_demoted():
    messages = build(40)
    messages[11] = {**messages[11], "governing": True}  # a mid-history standing instruction
    t = tier(active_tokens=200)
    t.step(messages, None, "q")
    assert t.demoted, "nothing was paged out; the test is not exercising the policy"
    assert not any(is_governing(messages[i]) for i in t.demoted)
    assert 0 not in t.demoted and 11 not in t.demoted


def test_last_k_messages_are_never_demoted():
    messages = build(40)
    t = tier(active_tokens=200, keep_last=6)
    t.step(messages, None, "q")
    assert t.demoted.isdisjoint(range(len(messages) - 6, len(messages)))


def test_demotion_prefers_the_oldest_first():
    messages = build(40)
    t = tier(active_tokens=600)
    t.step(messages, None, "q")
    # demoted indices form a contiguous run from the front (index 0 is governing)
    assert sorted(t.demoted) == list(range(1, 1 + len(t.demoted)))


# ------------------------------------------------------------ round-trip fidelity


def test_promotion_restores_the_original_bytes_exactly():
    messages = build(40)
    reference = canonical_bytes(messages)
    t = tier(active_tokens=2000)  # roomy enough that some demoted messages keep a stub
    t.step(messages, None, "q")
    demoted = sorted(t.demoted)
    assert demoted and t.stubbed()
    # the view is lossy: stubs stand in for some messages, others are gone entirely
    view = t.view(messages)
    assert canonical_bytes(view) != reference
    assert len(view) == len(messages) - len(t.evicted)
    shown = {
        int(m["content"].split("#")[1].split()[0])
        for m in view
        if m["content"].startswith("[cold #")
    }
    assert shown == t.stubbed()
    # ... but the messages behind it are untouched, and promoting is exact
    t.page(messages, {i: ("test", None) for i in demoted})
    assert t.demoted == set() and t.evicted == set()
    assert canonical_bytes(t.view(messages)) == reference


def test_view_preserves_role_and_governing_flag():
    messages = build(30)
    t = tier(active_tokens=200)
    t.step(messages, None, "q")
    kept = [m for i, m in enumerate(messages) if i not in t.demoted or i in t.stubbed()]
    for original, shown in zip(kept, t.view(messages), strict=True):
        assert shown["role"] == original["role"]
        assert shown.get("governing") == original.get("governing")


def test_demotion_reads_as_a_plain_content_edit_to_the_reuse_plan():
    from marathon import reuse_plan

    messages = build(30)
    t = tier(active_tokens=400)
    full = t.active_state(messages)  # nothing demoted yet
    t.step(messages, None, "q")
    paged = t.active_state(messages)
    plan = reuse_plan.plan(full, paged, lambda line: [0] * (len(line) // 4))
    assert plan.policy == "reuse", plan.reason  # not "repair": no governing span touched
    assert plan.segments  # the untouched tail is reusable, shifted


# -------------------------------------------------------------- recall triggers


def test_exact_trigger_promotes_a_demoted_message_the_delta_touched():
    messages = build(40)
    t = tier(active_tokens=300)
    t.step(messages, None, "q")
    target = sorted(t.demoted)[0]
    edited = [dict(m) for m in messages]
    fixed = edited[target]["content"] + " CORRECTION: 9-RHO."
    edited[target] = {**edited[target], "content": fixed}
    assert t.touched(messages, edited) == {target}
    events = t.step(edited, messages, "unrelated question")
    promoted = [e for e in events if e.kind == "promote"]
    assert target in [e.index for e in promoted]
    assert "exact" in next(e for e in promoted if e.index == target).reason
    assert target not in t.demoted
    assert edited[target]["content"] in [m["content"] for m in t.view(edited)]


def test_exact_trigger_ignores_untouched_and_still_hot_messages():
    messages = build(40)
    t = tier(active_tokens=300)
    t.step(messages, None, "q")
    hot = max(i for i in range(len(messages)) if i not in t.demoted)
    edited = [dict(m) for m in messages]
    edited[hot] = {**edited[hot], "content": "rewritten"}
    assert t.touched(messages, edited) == set()  # a hot message needs no recall


def test_query_trigger_recalls_the_message_that_holds_the_fact():
    messages = build(40)
    t = tier(active_tokens=300, top_k=2, threshold=0.05, embedder=HashEmbedder())
    t.step(messages, None, "q")
    target = sorted(t.demoted)[3]
    marker = messages[target]["content"].split()[-1]  # a token unique to that message
    hits = t.retrieve(messages, f"What was {marker} about?")
    assert hits and hits[0][0] == target
    events = t.step(messages, messages, f"What was {marker} about?")
    assert target in [e.index for e in events if e.kind == "promote"]


def test_query_trigger_respects_top_k_and_threshold():
    messages = build(40)
    t = tier(active_tokens=300, top_k=2, threshold=0.05)
    t.step(messages, None, "q")
    assert len(t.retrieve(messages, "entry0word1 entry1word1 entry2word1 entry3word1")) <= 2
    t.threshold = 0.999
    assert t.retrieve(messages, "completely unrelated vocabulary here") == []


def test_recall_disabled_never_promotes():
    messages = build(40)
    t = tier(active_tokens=300, recall=False)
    t.step(messages, None, "q")
    target = sorted(t.demoted)[0]
    edited = [dict(m) for m in messages]
    edited[target] = {**edited[target], "content": "totally different text"}
    events = t.step(edited, messages, edited[target]["content"])
    assert not [e for e in events if e.kind == "promote"]
    assert target in t.demoted


def test_a_promotion_that_overflows_the_window_demotes_something_else():
    messages = build(40)
    # a budget with slack, so there is still something eligible left to demote
    t = tier(active_tokens=4000)
    t.step(messages, None, "q")
    before = set(t.demoted)
    assert before and len(before) < len(messages) - 10
    target = sorted(before)[0]
    events = t.page(messages, {target: ("forced", 1.0)})
    assert target not in t.demoted  # the recall stands ...
    assert [e.index for e in events if e.kind == "demote"]  # ... paid for elsewhere
    assert t.active_token_count(messages) <= t.active_tokens + max(t.count(m) for m in messages)


def test_every_decision_is_logged_with_a_reason():
    messages = build(40)
    t = tier(active_tokens=300)
    t.step(messages, None, "q")
    t.page(messages, {sorted(t.demoted)[0]: ("forced", 0.9)})
    assert t.events and all(e.reason for e in t.events)
    assert {e.kind for e in t.events} == {"demote", "promote"}
    assert json.dumps([e.as_dict() for e in t.events])  # serializable for the metrics


# --------------------------------------------------------------- session wiring


def test_paging_a_real_session_keeps_replay_and_hashes_exact():
    session = Session()
    session.turn("system", "You are a careful assistant.")
    for i in range(20):
        session.turn("user", f"Entry {i}. " + " ".join(f"w{i}x{j}" for j in range(40)))
        session.turn("assistant", f"Noted {i}.")
    t = tier(active_tokens=400)
    t.step(session.messages, None, "carry on")
    assert t.demoted
    # the ledger's full history is untouched by paging
    assert session.replay() == canonical_bytes_lines(session.messages)
    # and the active view is a valid, decodable history
    decoded = Session.decode(t.active_state(session.messages))
    assert len(decoded) == len(t.view(session.messages))
    assert decoded[0] == session.messages[0]  # the system prompt is never paged out


def canonical_bytes_lines(messages: list[dict]) -> bytes:
    return b"".join(canonical_bytes(m) + b"\n" for m in messages)


def test_json_round_trip_of_the_view():
    messages = build(20)
    t = tier(active_tokens=300)
    t.step(messages, None, "q")
    state = t.active_state(messages)
    assert [json.loads(line) for line in state.split(b"\n") if line] == t.view(messages)


def test_chunking_finds_a_fact_buried_at_the_end_of_a_long_message():
    """Whole-message pooling loses one sentence in hundreds of words; chunking keeps it."""
    from marathon.cold import _chunks

    filler = " ".join(f"unrelated{w} padding{w}" for w in range(300))
    messages = [
        {"role": "system", "content": "sys", "governing": True},
        *[{"role": "user", "content": f"Entry {i}. {filler}"} for i in range(8)],
        {"role": "user", "content": f"Entry 8. {filler}\nThe zephyr code is 4821-OMEGA."},
        *[{"role": "user", "content": f"Entry {i}. {filler}"} for i in range(9, 14)],
    ]
    target = 9
    assert "4821-OMEGA" in messages[target]["content"]
    assert len(_chunks(messages[target]["content"])) > 1
    # the fact survives into exactly one chunk rather than being averaged away
    assert any("zephyr" in c for c in _chunks(messages[target]["content"]))

    t = ColdTier(active_tokens=500, top_k=2, threshold=0.05, keep_last=2)
    t.step(messages, None, "carry on")
    assert target in t.demoted
    hits = t.retrieve(messages, "What is the zephyr code?")
    assert hits and hits[0][0] == target


def test_chunks_cover_the_whole_message_with_overlap():
    from marathon.cold import _chunks

    words = [f"w{i}" for i in range(200)]
    chunks = _chunks(" ".join(words), words=60, overlap=20)
    assert len(chunks) > 3
    joined = " ".join(chunks)
    assert all(w in joined for w in words)  # nothing is dropped
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]  # they really overlap
