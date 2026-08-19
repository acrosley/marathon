"""From a state delta to a KV reuse plan — including when *not* to reuse.

``kvshift`` establishes that position-shifted reuse (keep ``P``, recompute ``E'``,
re-rotate ``S`` by ``delta``) is exact for position and approximate for attention:
``S``'s KV was computed attending to ``E``, not ``E'``. The 2026-08-18 findings entry
measured where that approximation actually costs an answer, and the split is sharp:

* **fact-carrying edits** (a value, a name, a code — even ones later text refers to by
  anaphora, even outright contradictions) survive plain reuse: the query attends to
  ``E'`` directly, so the fact only has to live in ``E'``, and ``S``'s stale states
  carry the *pointer*, which the edit did not move.
* **governing edits** — inside the system prompt, standing instructions, persona,
  output-format/language directives, tool policy — break it: first-token KL rose 30x
  and the model answered in a different language than full recompute.

So the classification the delta engine owes the KV layer is not "which spans mention
the edit" but "did the edit land in a span that governs later generation". It already
knows the byte offsets of every edit, and :func:`marathon.session.Session` now marks
governing messages, so the plan is a lookup, not an analysis.

The plan is a *list of segments*, not one span. An agent turn that rewrites four
messages, or moves a paragraph, leaves several unchanged runs, each with its own
``delta = dst_start - src_start`` (negative for a block that moved earlier). Matching is
done on canonical JSONL lines -- the delta engine's own unit of history -- so a segment
is always a whole run of messages and never needs snapping to a token boundary.

Policies emitted here: ``reuse`` (stitch and go), ``repair`` (stitch, but recompute the
first ``repair_first`` tokens of each non-leading segment natively so they attend to the
new text before them), ``full`` (no reuse). ``to_kv_transfer_params`` returns one dict
per reused segment, in destination order -- which is exactly one vLLM request each,
since the connector API can only express matched tokens as a prefix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .kvshift import Segment

# Findings 2026-08-18: first-32 already cut the governing-edit first-token KL ~10x and
# first-512 (10% of S) another 2x. 256 is the middle of that measured range.
DEFAULT_REPAIR_FIRST = 256


@dataclass(frozen=True)
class ReusePlan:
    """What the KV layer should do about one state transition.

    ``segments`` are the reused runs in destination order, each carrying its own
    ``delta``; ``total`` is the length of the new sequence in the same token
    coordinates. Everything the segments do not cover is recomputed. ``policy`` is one
    of {reuse, repair, full}; ``repair_first`` is the number of leading tokens of each
    non-leading segment to recompute natively (0 unless policy is ``repair``).

    The single-span vocabulary of the earlier design survives as derived properties:
    ``p`` (leading prefix), ``e_start``/``e_end`` (the first recomputed span), and
    ``s_start``/``s_end``/``delta`` (the last reused segment).
    """

    segments: tuple[Segment, ...]
    total: int
    policy: str
    repair_first: int
    reason: str

    @property
    def p(self) -> int:
        """Leading unchanged prefix, in tokens (0 if the very first token changed)."""
        lead = self.segments[0] if self.segments else None
        return lead.length if lead is not None and lead.dst_start == 0 and lead.delta == 0 else 0

    @property
    def e_start(self) -> int:
        return self.p

    @property
    def e_end(self) -> int:
        """End of the first recomputed span: the next segment, or the whole tail."""
        for seg in self.segments:
            if seg.dst_start > self.p:
                return seg.dst_start
        return self.total

    @property
    def _last(self) -> Segment | None:
        """The last reused segment that is not the leading prefix, if any."""
        last = self.segments[-1] if self.segments else None
        return last if last is not None and last.dst_start > 0 else None

    @property
    def delta(self) -> int:
        return self._last.delta if self._last else 0

    @property
    def s_start(self) -> int:
        return self._last.dst_start if self._last else self.total

    @property
    def s_end(self) -> int:
        return self._last.dst_end if self._last else self.total

    def to_kv_transfer_params(self) -> list[dict[str, int]]:
        """One ``{"dst_start", "dst_end", "delta"}`` per reused segment, in dst order.

        The leading prefix is omitted: vLLM's own prefix cache already has it. Every
        other segment needs the connector even at delta 0, because a recomputed span
        before it has already broken the prefix. ``dst_start`` skips the repaired head.
        Block alignment stays with the caller -- vLLM counts matched tokens in whole
        blocks, so ``local_probe`` rounds each ``dst_start`` up to a block boundary.
        """
        if self.policy == "full":
            return []
        out = []
        for seg in self.segments:
            if seg.dst_start == 0:
                continue
            start = seg.dst_start + self.repair_first
            if start < seg.dst_end:
                out.append({"dst_start": start, "dst_end": seg.dst_end, "delta": seg.delta})
        return out


def _lines(state: bytes) -> list[bytes]:
    """Canonical JSONL entries, newline included, as ``serialize_history`` wrote them."""
    return [line + b"\n" for line in state.split(b"\n") if line]


def _governing(line: bytes) -> bool:
    entry = json.loads(line)
    return bool(entry.get("governing", entry.get("role") == "system"))


def _match(old_lines: list[bytes], new_lines: list[bytes]) -> list[int | None]:
    """For each new line, the old line it reuses -- or None if it is genuinely new.

    Greedy and order-agnostic on purpose: a line that moved is still the same line, so
    the match may run backwards. Continuing the current run is preferred, which keeps a
    block of moved messages as one segment instead of many.
    """
    index: dict[bytes, list[int]] = {}
    for i, line in enumerate(old_lines):
        index.setdefault(line, []).append(i)
    used: set[int] = set()
    out: list[int | None] = []
    nxt: int | None = None
    for line in new_lines:
        if nxt is not None and nxt < len(old_lines) and nxt not in used and old_lines[nxt] == line:
            pick: int | None = nxt
        else:
            pick = next((i for i in index.get(line, ()) if i not in used), None)
        if pick is not None:
            used.add(pick)
        out.append(pick)
        nxt = None if pick is None else pick + 1
    return out


def _segments(
    matches: list[int | None],
    old_len: list[int],
    new_len: list[int],
    head_tokens: int,
) -> list[Segment]:
    """Maximal runs of consecutively-matched lines, as token-coordinate segments."""
    old_off, new_off = [head_tokens], [head_tokens]
    for n in old_len:
        old_off.append(old_off[-1] + n)
    for n in new_len:
        new_off.append(new_off[-1] + n)

    segs: list[Segment] = []
    if head_tokens:
        segs.append(Segment(0, head_tokens, 0))
    run: list[int] = []  # new-line indices in the current run

    def flush() -> None:
        if not run:
            return
        i0, i1 = matches[run[0]], matches[run[-1]]
        segs.append(Segment(old_off[i0], old_off[i1 + 1], new_off[run[0]]))
        run.clear()

    for j, m in enumerate(matches):
        if m is None:
            flush()
        elif run and matches[run[-1]] + 1 == m:
            run.append(j)
        else:
            flush()
            run.append(j)
    flush()

    merged: list[Segment] = []
    for seg in segs:
        if seg.length <= 0:
            continue
        prev = merged[-1] if merged else None
        if prev is not None and prev.dst_end == seg.dst_start and prev.src_end == seg.src_start:
            merged[-1] = Segment(prev.src_start, seg.src_end, prev.dst_start)
        else:
            merged.append(seg)
    return merged


def plan(
    old_state: bytes,
    new_state: bytes,
    tokenize,
    repair_first: int = DEFAULT_REPAIR_FIRST,
    head_tokens: int = 0,
) -> ReusePlan:
    """Classify the edits between two canonical states and size the reusable segments.

    ``tokenize`` maps one canonical JSONL line (bytes, trailing newline included) to the
    token ids that line contributes to the prompt; ``head_tokens`` is the length of
    anything the prompt puts *before* the first line (a system preamble). Token
    coordinates are therefore the serving layer's, not an abstract count.
    """
    old_lines, new_lines = _lines(old_state), _lines(new_state)
    old_len = [len(tokenize(line)) for line in old_lines]
    new_len = [len(tokenize(line)) for line in new_lines]
    total = head_tokens + sum(new_len)

    matches = _match(old_lines, new_lines)
    segments = _segments(matches, old_len, new_len, head_tokens)
    matched = {m for m in matches if m is not None}
    dropped = [i for i in range(len(old_lines)) if i not in matched]
    n_fresh = sum(1 for m in matches if m is None)
    moved = sum(1 for a, b in zip(segments, segments[1:], strict=False) if a.delta != b.delta)

    if not segments:
        # nothing at all survived: the history was replaced, not edited.
        return ReusePlan((), total, "full", 0, f"no reusable entries ({len(old_lines)} dropped)")

    what = (
        "append-only"
        if not dropped
        else (
            f"{len(dropped)} edited entries, {n_fresh} fresh, {len(segments)} segments"
            + (f", {moved} delta changes (moved blocks)" if moved else "")
        )
    )
    if any(_governing(old_lines[i]) for i in dropped):
        return ReusePlan(
            tuple(segments),
            total,
            "repair",
            repair_first,
            f"edit inside a governing span ({what}); reused KV must attend to the new "
            "text (findings 2026-08-18)",
        )
    return ReusePlan(tuple(segments), total, "reuse", 0, what)
