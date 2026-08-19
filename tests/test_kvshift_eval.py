"""CPU tests for the eval's session/edit construction (no model, no GPU).

What is worth proving here is the thing the whole eval rests on: every edit kind must
produce exactly *one* contiguous changed span between the old and new canonical states,
because a second disjoint edit would silently make ``token_span`` overstate ``E'`` and
quietly turn the measurement into something else.
"""

from __future__ import annotations

import pytest

from marathon.kvshift import byte_span, token_span
from marathon.kvshift_eval import EDIT_KINDS, FAMILIES, SNAPSHOT, build_item, load_corpus


def _byte_tokens(data: bytes) -> list[int]:
    """A stand-in tokenizer: one 'token' per byte. Span structure does not need a model."""
    return list(data)


@pytest.fixture(scope="module")
def corpus():
    """The frozen sample, so editing repo source cannot change what these assert."""
    return load_corpus(SNAPSHOT)


def test_live_corpus_matches_the_snapshot_shape():
    """The snapshot is only useful if the real loader still produces the same shape."""
    live, snap = load_corpus(), load_corpus(SNAPSHOT)
    assert live.keys() == snap.keys()
    for key in live:
        assert live[key], f"live corpus has no {key} entries"
        assert all(isinstance(x, tuple) and len(x) == 2 for x in live[key])
        assert all(len(text) > 400 for _, text in live[key])


@pytest.mark.parametrize("kind", EDIT_KINDS)
def test_edit_is_one_contiguous_span(kind, corpus):
    item = build_item(0, kind, "prose", corpus, seed=7, min_tokens=800, max_tokens=1200)
    old = item.session.replay()
    item.session.edit(item.msg_index, item.new_content)
    new = item.session.replay()
    assert old != new, f"{kind} edit changed nothing"

    head, tail = byte_span(old, new)
    # one span means the untouched head and tail account for everything but the edit
    assert head + tail <= len(new)
    span = token_span(_byte_tokens(old), _byte_tokens(new))
    assert span.p + span.e_old + span.s == len(old)
    assert span.p + span.e_new + span.s == len(new)
    # the changed region must lie inside a single message, not straddle several
    changed = new[span.p : span.p + span.e_new]
    assert changed.count(b"\n") <= 1, f"{kind} edit straddles canonical entries"


def test_governing_edit_lands_in_a_governing_message(corpus):
    item = build_item(1, "governing", "qa", corpus, seed=7, min_tokens=800, max_tokens=1200)
    assert item.msg_index == 0
    assert item.session.messages[0]["role"] == "system"
    assert item.session.messages[0].get("governing") is True


@pytest.mark.parametrize("kind", [k for k in EDIT_KINDS if k != "governing"])
def test_non_governing_edit_lands_in_a_user_turn(kind, corpus):
    item = build_item(2, kind, "code", corpus, seed=7, min_tokens=800, max_tokens=1200)
    assert item.session.messages[item.msg_index]["role"] == "user"
    assert not item.session.messages[item.msg_index].get("governing")


def test_delta_signs_match_the_edit_kind(corpus):
    """insert grows, delete shrinks, fact barely moves; rewrite may go either way."""
    signs = {"insert": 1, "delete": -1, "fact": 0}
    for kind, want in signs.items():
        item = build_item(3, kind, "qa", corpus, seed=11, min_tokens=800, max_tokens=1200)
        old = item.session.replay()
        item.session.edit(item.msg_index, item.new_content)
        delta = token_span(_byte_tokens(old), _byte_tokens(item.session.replay())).delta
        if want == 0:
            assert abs(delta) <= 8, (kind, delta)
        else:
            assert delta * want > 0, (kind, delta)


def test_fact_edit_changes_the_answer_and_others_do_not(corpus):
    fact = build_item(4, "fact", "prose", corpus, seed=5, min_tokens=800, max_tokens=1200)
    assert fact.facts["at_new"] != fact.facts["at"][1]
    at_query = next(q for q in fact.queries if q[0] == "fact-at")
    assert at_query[1] == [fact.facts["at_new"]]
    assert fact.facts["at_new"] in fact.new_content
    for kind in ("insert", "delete", "rewrite"):
        item = build_item(4, kind, "prose", corpus, seed=5, min_tokens=800, max_tokens=1200)
        assert "at_new" not in item.facts
        at_query = next(q for q in item.queries if q[0] == "fact-at")
        assert at_query[1] == [item.facts["at"][1]]
        assert item.facts["at"][1] in item.new_content, kind


def test_planted_facts_are_ordered_before_at_after(corpus):
    for family in FAMILIES:
        item = build_item(5, "insert", family, corpus, seed=3, min_tokens=800, max_tokens=1200)
        texts = [m["content"] for m in item.session.messages]
        where = {
            k: next(i for i, t in enumerate(texts) if item.facts[k][1] in t)
            for k in ("before", "at", "after")
        }
        assert where["before"] < where["at"] < where["after"]
        assert where["at"] == item.msg_index
        assert len({item.facts[k][0] for k in ("before", "at", "after")}) == 3


def test_build_is_deterministic_under_seed(corpus):
    a = build_item(9, "rewrite", "code", corpus, seed=42, min_tokens=800, max_tokens=1200)
    b = build_item(9, "rewrite", "code", corpus, seed=42, min_tokens=800, max_tokens=1200)
    assert a.session.replay() == b.session.replay()
    assert (a.new_content, a.queries) == (b.new_content, b.queries)
    c = build_item(9, "rewrite", "code", corpus, seed=43, min_tokens=800, max_tokens=1200)
    assert c.session.replay() != a.session.replay()
