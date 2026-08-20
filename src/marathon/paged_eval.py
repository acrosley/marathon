"""The paged eval population: sessions shaped like the cold tier's real failing workload.

Phase 3's population so far has been :mod:`marathon.kvshift_eval`'s single-span edits —
one contiguous rewrite in an otherwise untouched history. Track L then measured that on
the *real* cold-tier workload, paged/stitched reuse turns are wrong 75-91% of the time
even at low churn, which is a far worse failure than anything the synthetic population
shows. Those two facts cannot both be about the same thing, and the gap is the shape of
the edit: a paged turn is not one rewrite, it is **a front demotion plus up to two
promotions**, i.e. several disjoint edits scattered through the history, each one shifting
everything after it.

This module builds that population as :class:`marathon.stitch_train.Example` objects, so
the powered protocol (paired per-item deltas, the reference-stability probe, bootstrap
intervals) applies to it unchanged. What it reuses:

* :func:`marathon.cold_eval.build_session` for the long history with facts planted from
  the earliest turns to the latest;
* :class:`marathon.cold.ColdTier` for the paging policy itself — the same ``view``,
  the same ``[cold #k hash8: ...]`` stubs, the same protected-set rules that ship;
* :func:`marathon.kvshift.token_segments` for the *multi-segment* reuse plan, because
  ``token_span`` collapses several disjoint edits into one span running from the first
  change to the last, which would throw away exactly the reuse this workload depends on.

The measurement stays HF-side like ``kvshift_eval``: the stitched cache is built by
``kvshift``, the teacher is a full recompute of the new view, so a number here is
comparable with every other number in this phase.

**What the item is.** One paging *transition*: the active view before the turn (``old``,
whose KV the serving path reuses) and after it (``new``). Between them the tier promotes
one or two demoted messages back to full text and demotes the oldest eligible message to
a stub. The question then asks for a fact planted in a message that sits **after** all of
that churn — so the fact's text is inside the reused suffix ``S``, and every token of it
was computed while the promoted messages were still stubs and the demoted one was still
full text. That is the stale-attention hazard in its natural habitat rather than staged.
"""

from __future__ import annotations

import random

import torch

from .cold import ColdTier
from .kvshift import token_segments
from .kvshift_eval import load_corpus, question_text, render

#: active-view budget, in tokens. Chosen to sit in the same 4-8k regime the rest of the
#: phase is measured in, so a difference between populations is about *shape* rather than
#: about length.
ACTIVE_TOKENS = 6000
#: turns per session. Long enough that the early facts are far outside the active window
#: and paging is doing real work, short enough to build quickly.
TURNS = 40
#: promotions the transition performs, matching the cold tier's ``DEFAULT_TOP_K``
PROMOTIONS = 2


def _plain(messages: list[dict]) -> list[dict]:
    """Role/content only — the chat template should never see our bookkeeping keys."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def build_paged_examples(
    tok,
    device,
    n_items: int,
    seed: int,
    turns: int = TURNS,
    active_tokens: int = ACTIVE_TOKENS,
    promotions: int = PROMOTIONS,
    corpus=None,
) -> list:
    """``n_items`` paging transitions, as ``Example`` objects for :mod:`stitch_train`.

    Imported lazily from ``stitch_train`` to keep the module graph acyclic — that module
    owns the ``Example`` shape and this one only produces it.
    """
    from .stitch_train import Example

    corpus = corpus if corpus is not None else load_corpus()

    def ntok(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    def count(message: dict) -> int:
        return ntok(str(message.get("content", ""))) + 8

    from .cold_eval import build_session

    out: list = []
    for sid in range(n_items):
        rng = random.Random((seed * 1_000_003) ^ (sid * 7919))
        sess = build_session(sid, corpus, seed, turns, count_tokens=ntok)
        tier = ColdTier(active_tokens=active_tokens, count=count)
        tier.page(sess.messages)  # settle into the paged steady state

        # The fact must be *resident* in the old view and sit after the churn, so its KV
        # is reused while conditioned on a prefix that no longer exists. On a long enough
        # session every planted fact has been paged out by now, so the target is recalled
        # first -- which is not a contrivance, it is precisely what the exact trigger does
        # when a user asks about demoted text. That recall is the *previous* turn; the
        # transition under test is the one after it.
        planted = [f for f in sess.facts if "msg_index" in f and f["msg_index"] > 4]
        if not planted:
            continue
        target = max(planted, key=lambda f: f["msg_index"])
        pin = target["msg_index"]
        tier.page(sess.messages, promote={pin: ("exact", None)})
        if pin in tier.demoted:  # the budget refused to keep it; this item cannot ask
            continue
        old_view = tier.view(sess.messages)

        # This turn: recall one or two more demoted messages that sit *before* the fact.
        # Each promotion is a grow edit in the middle of the history and each shifts
        # everything after it, while the budget pays for them by demoting the oldest
        # eligible message -- a shrink at the front. Several disjoint edits, one turn.
        candidates = sorted(i for i in tier.demoted if i < pin)
        if not candidates:
            continue
        picks = candidates[-promotions:] if len(candidates) > promotions else candidates
        rng.shuffle(picks)
        promote = {i: ("query", 0.5) for i in picks}
        promote[pin] = ("pin", None)  # keep the fact resident through this turn's churn
        tier.page(sess.messages, promote=promote)
        if pin in tier.demoted:
            continue
        new_view = tier.view(sess.messages)

        old_ids = tok.encode(render(tok, _plain(old_view)), add_special_tokens=False)
        new_ids = tok.encode(render(tok, _plain(new_view)), add_special_tokens=False)
        segments = token_segments(old_ids, new_ids)
        if not segments:
            continue
        # A reusable suffix must actually reach the end of the new sequence, or there is
        # no stale span for the model to be wrong about and the item tests nothing.
        if segments[-1].dst_end < len(new_ids) - 1 or len(segments) < 2:
            continue

        question = f"What is the {target['noun']} code?"
        forced = f" The {target['noun']} code is"
        out.append(
            Example(
                sid=sid,
                edit_kind="paged",
                family="paged",
                old_ids=torch.tensor(old_ids, device=device),
                new_ids=torch.tensor(new_ids, device=device),
                query_ids=torch.tensor(
                    tok.encode(
                        question_text(tok, _plain(new_view), question, forced),
                        add_special_tokens=False,
                    ),
                    device=device,
                ),
                qtype="paged-fact",
                expected=[target["code"]],
                span=segments,
                meta={
                    "segments": len(segments),
                    "promoted": len(picks),
                    "target_index": pin,
                    "demoted": len(tier.demoted),
                    "old_tokens": len(old_ids),
                    "new_tokens": len(new_ids),
                    "reused": sum(s.src_end - s.src_start for s in segments),
                },
            )
        )
    return out
