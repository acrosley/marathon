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

Policies emitted here: ``reuse`` (stitch and go), ``repair`` (stitch, but recompute the
first ``repair_first`` tokens of ``S`` natively so they attend to ``E'``), ``full`` (no
reuse). ``to_kv_transfer_params`` returns exactly the dict the vLLM shift connector
takes, so a probe can adopt this in one line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .kvshift import Span, byte_span, token_span

# Findings 2026-08-18: first-32 already cut the governing-edit first-token KL ~10x and
# first-512 (10% of S) another 2x. 256 is the middle of that measured range.
DEFAULT_REPAIR_FIRST = 256


@dataclass(frozen=True)
class ReusePlan:
    """What the KV layer should do about one state transition.

    ``p`` unchanged prefix (tokens). ``e_start``/``e_end`` the recomputed span ``E'`` in
    *new* token coordinates (for an append-only turn this is the appended tail).
    ``delta`` the position shift applied to ``S``. ``s_start``/``s_end`` the reusable
    span in new coordinates. ``policy`` in {reuse, repair, full}; ``repair_first`` is the
    number of leading ``S`` tokens to recompute natively (0 unless policy is ``repair``).
    """

    p: int
    e_start: int
    e_end: int
    delta: int
    s_start: int
    s_end: int
    policy: str
    repair_first: int
    reason: str

    def to_kv_transfer_params(self) -> dict[str, int] | None:
        """``{"dst_start", "dst_end", "delta"}`` for ``vllm_shift_connector``, or None.

        ``dst_start`` skips the repaired head, which the caller prefills natively.
        None means "do not reuse" (policy ``full``, or nothing left to copy).
        """
        start = self.s_start + self.repair_first
        if self.policy == "full" or start >= self.s_end:
            return None
        # ponytail: block alignment is the caller's problem — vLLM counts matched tokens
        # in whole blocks, so local_probe rounds dst_start up to a block boundary.
        return {"dst_start": start, "dst_end": self.s_end, "delta": self.delta}


def _lines(state: bytes) -> list[bytes]:
    """Canonical JSONL entries, newline included, as ``serialize_history`` wrote them."""
    return [line + b"\n" for line in state.split(b"\n") if line]


def _governing(line: bytes) -> bool:
    entry = json.loads(line)
    return bool(entry.get("governing", entry.get("role") == "system"))


def _line_at(lines: list[bytes], offset: int) -> int:
    """Index of the entry containing byte ``offset`` (clamped to the last entry)."""
    pos = 0
    for i, line in enumerate(lines):
        pos += len(line)
        if offset < pos:
            return i
    return len(lines) - 1


def plan(
    old_state: bytes,
    new_state: bytes,
    tokenize,
    repair_first: int = DEFAULT_REPAIR_FIRST,
) -> ReusePlan:
    """Classify the edit between two canonical states and size the KV spans.

    ``tokenize`` maps ``bytes`` to a list of token ids. Appended entries are separated
    off first (they are the new turn, never an "edit"), so ``S`` is the run of unchanged
    history *between* the edit and the new turn — the same span ``local_probe`` reuses.
    """
    old_lines, new_lines = _lines(old_state), _lines(new_state)
    kept = new_lines[: len(old_lines)]
    edited = [i for i, (a, b) in enumerate(zip(old_lines, kept, strict=False)) if a != b]
    new_ids = tokenize(new_state)

    if not edited and len(kept) == len(old_lines):
        # append-only (or no change at all): E' is the appended tail, S is empty.
        p = len(tokenize(old_state))
        return ReusePlan(
            p, p, len(new_ids), 0, len(new_ids), len(new_ids), "reuse", 0, "append-only"
        )

    if not edited or len(edited) > 1:
        # ponytail: one shifted span is what stitch()/the connector can express. Multiple
        # disjoint edits need a multi-span plan (per-span delta); until then, recompute.
        return ReusePlan(
            0,
            0,
            len(new_ids),
            0,
            len(new_ids),
            len(new_ids),
            "full",
            0,
            f"{len(edited)} disjoint edits or truncated history; multi-span reuse not implemented",
        )

    core = b"".join(kept)  # new history minus the appended turn
    head_bytes, _ = byte_span(old_state, core)  # delta engine's view of where it changed
    span: Span = token_span(tokenize(old_state), tokenize(core))
    s_start = span.p + span.e_new
    s_end = s_start + span.s

    if _governing(old_lines[_line_at(old_lines, head_bytes)]):
        return ReusePlan(
            span.p,
            span.p,
            s_start,
            span.delta,
            s_start,
            s_end,
            "repair",
            min(repair_first, span.s),
            "edit inside a governing span: S must attend to E' (findings 2026-08-18)",
        )
    return ReusePlan(
        span.p,
        span.p,
        s_start,
        span.delta,
        s_start,
        s_end,
        "reuse",
        0,
        "fact-carrying edit in a non-governing span",
    )
