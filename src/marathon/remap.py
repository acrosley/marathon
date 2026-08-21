"""Keep reused KV at generation 0: a logical-position -> store-index map.

Today an edit turn saves with ``"full"``, re-gathering the whole prompt out of the paged
cache -- including the span the connector just stitched there. So the store's copy of a
reused span is a *rotated copy of a rotated copy* after two reuse turns, and by
construction it is indistinguishable from freshly computed KV, which means the next turn
reuses it believing it is fresh. Staleness compounds silently. That is the compounding
hypothesis for the paged workload's answer collapse (findings 2026-08-21).

The alternative is not to re-save reused spans at all. The store already holds the
original bytes; what it lacks is a way to *address* them once the text they belong to has
moved. This is that address book.

The invariant is unchanged and is what makes the whole thing work: **store index ``i``
holds keys as computed at position ``i``**. So a span living at store ``i`` that now
belongs at logical ``p`` needs a re-rotation of exactly ``p - i`` -- which is the offset
recorded here. Offsets compose by addition, because rotations do: a span shifted by
``d1`` and later by ``d2`` is at offset ``d1 + d2`` from its generation-0 index, and one
rotation by that sum is exact. So reuse can go on indefinitely without ever stitching
from stitched bytes.

Freshly computed positions are saved at their own logical index and therefore have
offset 0, which is why an append-only session never accumulates anything here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: ``(lo, hi, offset)``: logical ``[lo, hi)`` lives at store ``[lo - offset, hi - offset)``
Piece = tuple[int, int, int]


@dataclass
class Remap:
    """Where each logical position's generation-0 KV actually lives in the store."""

    pieces: list[Piece] = field(default_factory=list)

    def translate(self, lo: int, hi: int) -> list[Piece]:
        """Split ``[lo, hi)`` into the store ranges that back it, in logical order.

        A span may cross several pieces once different parts of the history have moved by
        different amounts, and each part needs its own re-rotation. Anything not covered
        is offset 0 -- freshly computed positions are stored where they belong.
        """
        out: list[Piece] = []
        cursor = lo
        for p_lo, p_hi, off in sorted(self.pieces):
            if p_hi <= cursor or p_lo >= hi:
                continue
            if p_lo > cursor:
                out.append((cursor, min(p_lo, hi), 0))
                cursor = min(p_lo, hi)
            end = min(p_hi, hi)
            if end > cursor:
                out.append((cursor, end, off))
                cursor = end
            if cursor >= hi:
                break
        if cursor < hi:
            out.append((cursor, hi, 0))
        return [(a, b, o) for a, b, o in out if b > a]

    def after_turn(self, segments: list[tuple[int, int, int]], total: int) -> Remap:
        """The map for the next turn, given this turn's reused ``(dst_lo, dst_hi, delta)``.

        A reused span's offset is its old offset plus this turn's delta -- the additive
        composition that keeps it pointing at generation 0. Everything the segments do
        not cover was computed and saved this turn, so it drops back to offset 0 and is
        simply absent from the new map.
        """
        out: list[Piece] = []
        for dst_lo, dst_hi, delta in sorted(segments):
            if dst_hi <= dst_lo:
                continue
            # where this span sat before the turn, so we can pick up its old offset
            for s_lo, s_hi, old in self.translate(dst_lo - delta, dst_hi - delta):
                out.append((s_lo + delta, s_hi + delta, old + delta))
        return Remap(_merge(out, total))

    def restrict(self, hi: int) -> Remap:
        """Keep only what lies below ``hi``.

        A turn that reused nothing recomputed and re-saved everything from the first
        change onward, so those positions are back at offset 0; the untouched leading
        prefix keeps whatever mapping it had, because neither its text nor its stored
        bytes moved.
        """
        return Remap(_merge([(lo, min(h, hi), off) for lo, h, off in self.pieces], hi))

    def offsets(self) -> set[int]:
        return {o for _, _, o in self.pieces}

    def depth(self) -> int:
        """Largest accumulated offset: how far any span has drifted from generation 0."""
        return max((abs(o) for _, _, o in self.pieces), default=0)


def _merge(pieces: list[Piece], total: int) -> list[Piece]:
    """Sort, clip to ``[0, total)``, drop empties and join neighbours sharing an offset."""
    out: list[Piece] = []
    for lo, hi, off in sorted(pieces):
        lo, hi = max(lo, 0), min(hi, total)
        if hi <= lo:
            continue
        if out and out[-1][1] == lo and out[-1][2] == off:
            out[-1] = (out[-1][0], hi, off)
        else:
            out.append((lo, hi, off))
    return out


def loads_for(remap: Remap, segments: list[dict]) -> list[dict]:
    """Turn plan segments into loads whose ``delta`` reaches generation-0 bytes.

    Without a remap a segment is one load at the plan's own delta. With one, a segment
    may split into several loads, each re-rotating by however far *that* part has drifted
    since it was computed. The connector needs no changes: it already reads
    ``src = dst_start - delta``.

    Coordinates, which are easy to get backwards: ``remap`` must be the map as it stood
    *before* this turn, because a plan's ``delta`` is measured against the previous
    state. The order per turn is therefore ``loads_for(remap, segments)`` first, then
    ``remap = remap.after_turn(segments, total)``.
    """
    out: list[dict] = []
    for seg in segments:
        lo, hi, delta = int(seg["dst_start"]), int(seg["dst_end"]), int(seg["delta"])
        for p_lo, p_hi, off in remap.translate(lo - delta, hi - delta):
            out.append({"dst_start": p_lo + delta, "dst_end": p_hi + delta, "delta": off + delta})
    return out
