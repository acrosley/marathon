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
    moved: tuple[bool, ...]
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

    def to_kv_transfer_params(self, reuse_moved: bool = False) -> list[dict[str, int]]:
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
        for seg, moved in zip(self.segments, self.moved, strict=True):
            if seg.dst_start == 0:
                continue
            if moved and not reuse_moved:
                # Measured 2026-08-19: transplanting a *relocated* block into vLLM at
                # |delta| ~10k destroys generation, and repairing its head does not help.
                # A shift is safe; a relocation is not. Recompute it instead.
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

    Order-agnostic on purpose: a line that moved is still the same line, so the match
    may run backwards. But byte identity is not *context* identity -- a reused line's KV
    encodes whatever preceded it in the old sequence -- and histories are full of
    byte-identical entries (every bare ``"ok"`` acknowledgement is the same line). Given
    several candidates, the nearest one is therefore taken, not the first: an entry that
    did not move should match itself, and only a genuinely relocated entry should get a
    large delta. Continuing the current run breaks ties, which keeps a block of moved
    entries together as one segment.

    Measured 2026-08-18: picking the first unused candidate instead mapped a duplicate
    acknowledgement line to one 10k tokens away and destroyed generation.
    """
    index: dict[bytes, list[int]] = {}
    for i, line in enumerate(old_lines):
        index.setdefault(line, []).append(i)
    used: set[int] = set()
    out: list[int | None] = []
    nxt: int | None = None
    for j, line in enumerate(new_lines):
        free = [i for i in index.get(line, ()) if i not in used]
        pick = min(free, key=lambda i: (abs(i - j), i != nxt)) if free else None
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
) -> tuple[list[Segment], list[bool]]:
    """Maximal runs of consecutively-matched lines, as token-coordinate segments.

    The second list flags the runs whose entries *relocated* (their index in the history
    changed), as opposed to merely shifting because something before them got longer.
    """
    old_off, new_off = [head_tokens], [head_tokens]
    for n in old_len:
        old_off.append(old_off[-1] + n)
    for n in new_len:
        new_off.append(new_off[-1] + n)

    segs: list[Segment] = []
    moved: list[bool] = []
    if head_tokens:
        segs.append(Segment(0, head_tokens, 0))
        moved.append(False)
    run: list[int] = []  # new-line indices in the current run

    def flush() -> None:
        if not run:
            return
        i0, i1 = matches[run[0]], matches[run[-1]]
        segs.append(Segment(old_off[i0], old_off[i1 + 1], new_off[run[0]]))
        moved.append(any(matches[j] != j for j in run))
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
    merged_moved: list[bool] = []
    for seg, mv in zip(segs, moved, strict=True):
        if seg.length <= 0:
            continue
        prev = merged[-1] if merged else None
        if prev is not None and prev.dst_end == seg.dst_start and prev.src_end == seg.src_start:
            merged[-1] = Segment(prev.src_start, seg.src_end, prev.dst_start)
            merged_moved[-1] = merged_moved[-1] or mv
        else:
            merged.append(seg)
            merged_moved.append(mv)
    return merged, merged_moved


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
    segments, moved = _segments(matches, old_len, new_len, head_tokens)
    matched = {m for m in matches if m is not None}
    dropped = [i for i in range(len(old_lines)) if i not in matched]
    n_fresh = sum(1 for m in matches if m is None)

    if not segments:
        # nothing at all survived: the history was replaced, not edited.
        return ReusePlan(
            (), (), total, "full", 0, f"no reusable entries ({len(old_lines)} dropped)"
        )

    relocated = [m for j, m in enumerate(matches) if m is not None and m != j]
    if not dropped and not relocated:
        what = "append-only"
    elif not dropped:
        what = f"{len(relocated)} relocated entries, {len(segments)} segments (moved blocks)"
    else:
        what = f"{len(dropped)} edited entries, {n_fresh} fresh, {len(segments)} segments" + (
            f", {len(relocated)} relocated" if relocated else ""
        )
    # A governing entry that *moved* changes what governs later text just as much as one
    # that was rewritten, so both count. An index change is the signal; a token delta can
    # cancel to zero while the entry still sits somewhere else in the history.
    if any(_governing(old_lines[i]) for i in dropped + relocated):
        return ReusePlan(
            tuple(segments),
            tuple(moved),
            total,
            "repair",
            repair_first,
            f"edit inside a governing span ({what}); reused KV must attend to the new "
            "text (findings 2026-08-18)",
        )
    return ReusePlan(tuple(segments), tuple(moved), total, "reuse", 0, what)


def phases(loads: list[dict], block_size: int, n_prompt: int) -> list[tuple[int, dict | None]]:
    """Turn a load list into ``[(request_length, load_or_None)]`` — k+1 requests.

    vLLM's connector API can only express externally-matched tokens as a *prefix* of
    one request, so k reused segments cannot be handed over in one shot. They are
    instead handed over one per request, in destination order: request i prefills the
    fresh span before segment i (everything earlier is a vLLM prefix hit, including the
    segments the connector wrote in earlier requests) and stops exactly on the block
    boundary where segment i begins, so that the *next* request's local prefix hit lands
    on the connector's ``dst_start``. The final request is the real one: full prompt,
    last segment loaded, only the new turn left to prefill.

    Segments are clipped to whole blocks (vLLM counts matched tokens per block); a
    segment with less than one whole block left is dropped and simply recomputed.
    An empty result means "one ordinary request, no connector loads".
    """
    segs = []
    for ld in loads:
        lo = -(-int(ld["dst_start"]) // block_size) * block_size
        hi = int(ld["dst_end"]) // block_size * block_size
        if hi - lo >= block_size and lo > 0 and hi < n_prompt:
            segs.append((lo, hi, int(ld["delta"])))
    if not segs:
        return []
    out: list[tuple[int, dict | None]] = [(segs[0][0], None)]
    for i, (lo, _hi, _d) in enumerate(segs[1:], start=1):
        prev = segs[i - 1]
        out.append((lo, {"dst_start": prev[0], "dst_end": prev[1], "delta": prev[2]}))
    last = segs[-1]
    out.append((n_prompt, {"dst_start": last[0], "dst_end": last[1], "delta": last[2]}))
    return out
