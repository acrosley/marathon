"""Turn a reuse plan into a non-prefix match mask: which positions the engine must compute.

vLLM's KV-connector API can only express externally supplied KV as a *prefix* of one
request (``get_num_new_matched_tokens`` returns a count, and the scheduler turns it into
a scalar ``num_computed_tokens``). That is why :func:`marathon.reuse_plan.phases` hands
``k`` reused segments over as ``k + 1`` sequential requests. Measured 2026-08-21, that
multi-request path is where the paged workload loses its answers: the stitched KV itself
is bit-right to bf16 in every layer, yet fact exact-match drops 0.75 -> 0.35 on Qwen3-8B,
and capping the plan at a single segment recovers a third of the gap.

The single-request alternative is to let the connector declare a *set* of already-filled
spans and have the engine prefill only the gaps between them. That is sound: attention
reads every earlier position out of the paged cache through the block table, so a token
does not have to be *computed* in the same forward pass as the tokens it attends to --
it only has to be *present*. The connector writes the reused spans in ``start_load_kv``,
which runs before the forward, and gaps computed earlier in the same batch are written to
their slots before the later gaps attend to them.

This module is the part of that design that needs no vLLM at all: given the loads a plan
emits, produce the block-aligned spans the connector will fill and the exact list of
positions the engine must still compute. ``scripts/patch_vllm_gapfill.py`` is the other
half. Everything here is pure and unit-tested; the invariant the patch relies on is
:func:`check`, which is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

Span = tuple[int, int]


@dataclass(frozen=True)
class GapPlan:
    """What the engine is told: ``filled`` spans it may skip, ``compute`` positions it must not.

    ``filled`` are block-aligned, sorted, disjoint and never touch the last token of the
    prompt -- the engine must always be left at least one position to compute, or there
    is nothing to run a forward pass on and no logits to sample from. ``compute`` is
    exactly the complement, so ``len(compute) + filled_tokens == n_prompt`` always.
    """

    filled: tuple[Span, ...]
    compute: tuple[int, ...]
    n_prompt: int

    @property
    def filled_tokens(self) -> int:
        return sum(hi - lo for lo, hi in self.filled)

    @property
    def saving(self) -> float:
        """Fraction of the prompt the engine does not have to prefill."""
        return self.filled_tokens / self.n_prompt if self.n_prompt else 0.0


def align(loads: list[dict], block_size: int, n_prompt: int) -> list[Span]:
    """Block-align the plan's loads and drop what cannot survive alignment.

    vLLM accounts for externally supplied KV in whole blocks, so a span is clipped
    inward to block boundaries; anything left with less than one whole block is dropped
    and simply recomputed. The last block of the prompt is never claimed, which is what
    guarantees the engine has something to compute.
    """
    limit = (n_prompt - 1) // block_size * block_size
    out: list[Span] = []
    for ld in loads:
        lo = -(-int(ld["dst_start"]) // block_size) * block_size
        hi = min(int(ld["dst_end"]), limit) // block_size * block_size
        if hi - lo >= block_size and lo >= 0:
            out.append((lo, hi))
    out.sort()
    merged: list[Span] = []
    for lo, hi in out:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def plan_gaps(loads: list[dict], block_size: int, n_prompt: int, local_hit: int = 0) -> GapPlan:
    """The full mask: block-aligned filled spans plus the positions still to compute.

    ``local_hit`` is what the engine's own prefix cache already holds, which counts as
    filled: those positions are computed and must not appear in ``compute``, or the
    request would be told to recompute a prefix it already has -- at positions the
    scheduler has not allocated it any budget for.
    """
    filled = align(loads, block_size, n_prompt)
    if local_hit > 0:
        filled = align(
            [{"dst_start": 0, "dst_end": min(local_hit, n_prompt)}]
            + [{"dst_start": lo, "dst_end": hi} for lo, hi in filled],
            block_size,
            n_prompt,
        )
    compute: list[int] = []
    cursor = 0
    for lo, hi in filled:
        compute.extend(range(cursor, lo))
        cursor = hi
    compute.extend(range(cursor, n_prompt))
    plan = GapPlan(tuple(filled), tuple(compute), n_prompt)
    check(plan)
    return plan


def check(plan: GapPlan) -> None:
    """The invariants the vLLM patch relies on. Cheap enough to assert every time."""
    prev = 0
    for lo, hi in plan.filled:
        assert 0 <= lo < hi <= plan.n_prompt, f"bad span {(lo, hi)} in {plan.n_prompt}"
        assert lo >= prev, f"spans out of order or overlapping at {(lo, hi)}"
        prev = hi
    assert plan.compute and plan.compute[-1] == plan.n_prompt - 1, (
        "the engine must always be left the final position to compute"
    )
    assert len(plan.compute) + plan.filled_tokens == plan.n_prompt, (
        f"{len(plan.compute)} computed + {plan.filled_tokens} filled != {plan.n_prompt}"
    )
    assert len(set(plan.compute)) == len(plan.compute), "duplicate compute positions"
