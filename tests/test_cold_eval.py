"""CPU tests for the Phase 2 eval's session construction and scoring.

No GPU and no model: what is checked here is that the eval *can* measure what it
claims to measure -- above all that a demoted message really hides its planted fact,
since a stub that leaks the answer would score recall the policy never performed.
"""

from __future__ import annotations

import pytest

from marathon.cold import ColdTier, stub_text
from marathon.cold_eval import build_session, drive, extract_code, score, summarise
from marathon.kvshift_eval import SNAPSHOT, load_corpus


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(SNAPSHOT)


def test_facts_are_planted_and_spread_across_the_history(corpus):
    item = build_session(0, corpus, 17, 40)
    assert len(item.facts) == 6
    turns = [f["turn"] for f in item.facts]
    assert turns == sorted(turns)
    assert turns[0] < 8 and turns[-1] > 30  # genuinely early and genuinely late
    for f in item.facts:
        assert f["code"] in item.messages[f["msg_index"]]["content"]


def test_a_demoted_message_does_not_leak_its_fact_through_the_stub(corpus):
    """The eval's load-bearing invariant: paging a message out really hides the answer.

    Regression test for a real bug in the first run of this eval -- the fact was planted
    in the opening words of the message, and the stub keeps the opening words, so the
    no-recall baseline scored 0.5 on questions it could read straight off the stub.
    """
    for sid in range(6):
        item = build_session(sid, corpus, 17, 40)
        for f in item.facts:
            index = f["msg_index"]
            assert f["code"] not in stub_text(index, item.messages[index])


def test_every_fact_is_hidden_by_the_active_view_once_paged_out(corpus):
    item = build_session(0, corpus, 17, 40)
    tier = ColdTier(active_tokens=2000)
    tier.step(item.messages, None, "carry on")
    hidden = [f for f in item.facts if f["msg_index"] in tier.demoted]
    assert hidden, "no fact was paged out; the budget is too generous to test anything"
    view = "".join(m["content"] for m in tier.view(item.messages))
    for f in hidden:
        assert f["code"] not in view


def test_questions_cover_old_recent_and_distractors(corpus):
    item = build_session(1, corpus, 17, 40)
    kinds = [q["kind"] for q in item.questions]
    assert set(kinds) == {"old", "recent", "distractor"}
    nouns = {f["noun"] for f in item.facts}
    for q in item.questions:
        if q["kind"] == "distractor":
            assert q["noun"] not in nouns and q["expected"] is None and q["target"] is None
        else:
            assert q["expected"] and q["target"] is not None


def test_sessions_are_deterministic(corpus):
    a = build_session(3, corpus, 17, 30)
    b = build_session(3, corpus, 17, 30)
    assert a.messages == b.messages and a.facts == b.facts
    assert build_session(4, corpus, 17, 30).messages != a.messages


def test_extract_code_and_scoring():
    assert extract_code("The answer is 4821-OMEGA.") == "4821-OMEGA"
    assert extract_code("4821-omega") == "4821-OMEGA"
    assert extract_code("not mentioned") is None
    assert extract_code("8109-BEACODE") is None  # a hallucinated shape is not a code
    fact = {"kind": "old", "expected": "4821-OMEGA"}
    assert score(fact, "4821-OMEGA")["correct"]
    assert not score(fact, "1111-ZETA")["correct"]
    dis = {"kind": "distractor", "expected": None}
    assert score(dis, "not mentioned")["correct"]
    assert score(dis, "It is 1111-ZETA.")["fabricated"]


def test_drive_and_summarise_end_to_end(corpus):
    """The whole eval loop against fakes: history turns, question turns, one table."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_server import CountingTokenizer, FakeEngine

    from marathon.server import MarathonServer

    item = build_session(0, corpus, 17, 16)
    rows = []
    for cond, kw in [("full", {}), ("cold-recall", {"active_window": 3000})]:
        server = MarathonServer(engine=FakeEngine(), tokenizer=CountingTokenizer(), **kw)
        for row in drive(server, f"{cond}-0", item):
            row.update(condition=cond, sid=0)
            rows.append(row)

    assert {r["phase"] for r in rows} == {"history", "question"}
    table = summarise(rows)
    assert [t["condition"] for t in table] == ["full", "cold-recall"]
    full, cold = table
    # the point of the whole exercise: the paged window is smaller and stays smaller
    assert cold["active_max"] < full["active_max"]
    assert all(t["turns"] == 17 for t in table)  # 16 user turns plus the system prompt
