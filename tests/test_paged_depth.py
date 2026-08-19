"""Depth regression: many *consecutive* front-of-view edits through the real store.

Phase 1 validated shifted KV reuse on one edit per session, and the ``save="full"``
rebuild on two and three. A *paged* session (Phase 2's cold tier) is a different animal:
demoting the oldest message every turn is a shrink edit at the **front** of the view on
every single turn, so a 70-turn session is 70 consecutive edit turns, each with a
negative delta and almost no leading prefix. Track N measured that shape breaking —
exact match 1.0 with the connector off against 0.33 with it on, degrading progressively.

This module reproduces that shape on CPU, with no vLLM and no GPU, by modelling "the KV
of the token at position p" as "the token id at position p". Re-rotation is exact by
construction in that model, so it cannot mask anything: a fingerprint that comes back
wrong means the *coordinates* were wrong — the connector handed vLLM KV for the wrong
token, and the real engine would have generated from it silently. The decisions under
test are the connector's own, imported from :mod:`marathon.shift_store` rather than
re-implemented here; the paged cache, block-level prefix cache and per-step save loop
around them mirror vLLM's.
"""

from __future__ import annotations

import pytest

from marathon import client as mclient
from marathon.server import MarathonServer

torch = pytest.importorskip("torch")

from marathon.shift_store import ShiftStore, plan_load, plan_save, slots  # noqa: E402

BLOCK = 16
FILLER = "deterministic filler that makes a message worth several blocks of tokens "


class FakeTokenizer:
    """One token per byte of the rendered message; distinct messages, distinct ids."""

    def prompt(self, messages):
        pieces = [list(f"{m['role']}: {m['content']}\n".encode()) for m in messages]
        return [i for p in pieces for i in p] + [10], pieces


class FingerprintEngine:
    """A paged KV cache whose 'KV' is the token id, driven by the real decisions.

    Models the parts of vLLM the connector actually depends on: block-granular prefix
    caching, a block table per request, ``num_computed_tokens`` advancing over an
    external match, and a save planned once per scheduler step — decode steps included,
    which is where the paged cost bug lives. Everything else is left out on purpose.
    """

    block_size = BLOCK

    def __init__(self, budget_tokens: int = 1 << 20):
        self.store = ShiftStore(budget_tokens, device="cpu", allocate=True)
        self.slots: list[int] = []  # the paged cache: slot -> token id (0 = never written)
        self.prefix: dict[str, int] = {}  # key of ids[:(k+1)*B] -> block id
        self.declines: list[str] = []
        self.refused_saves = 0
        self.corruptions: list[tuple[int, int, int, int]] = []  # (turn, pos, want, got)
        self.saved_token_steps = 0  # positions written to the store, summed over steps
        self.turn = -1

    # ---------------------------------------------------------------- paged cache

    def _new_block(self) -> int:
        self.slots.extend([0] * BLOCK)
        return len(self.slots) // BLOCK - 1

    def _key(self, ids: list[int], k: int) -> str:
        return str(ids[: (k + 1) * BLOCK])

    def _blocks_for(self, ids: list[int]) -> tuple[list[int], int]:
        """Block table for ``ids`` plus the block-aligned prefix hit, as vLLM would."""
        blocks, hit, broken = [], 0, False
        for k in range((len(ids) + BLOCK - 1) // BLOCK):
            cached = None if broken else self.prefix.get(self._key(ids, k))
            if cached is not None and (k + 1) * BLOCK <= len(ids):
                blocks.append(cached)
                hit = (k + 1) * BLOCK
            else:
                broken = True
                blocks.append(self._new_block())
        return blocks, hit

    def _publish(self, ids: list[int], blocks: list[int]) -> None:
        for k in range(len(ids) // BLOCK):
            self.prefix.setdefault(self._key(ids, k), blocks[k])

    def _write(self, blocks: list[int], lo: int, hi: int, values: list[int]) -> None:
        for slot, v in zip(slots(blocks, lo, hi, BLOCK).tolist(), values, strict=True):
            self.slots[slot] = v

    def _read(self, blocks: list[int], lo: int, hi: int) -> list[int]:
        return [self.slots[s] for s in slots(blocks, lo, hi, BLOCK).tolist()]

    # -------------------------------------------------------------------- the API

    def generate(self, ids, session, max_tokens, load=None, save=False):
        blocks, num_computed = self._blocks_for(ids)
        params: dict = {"save": save}

        decision, why = plan_load(self.store, session, load, num_computed, len(ids), BLOCK)
        if load and decision is None:
            self.declines.append(f"turn {self.turn}: {why}")
        if decision is not None:
            # the worker's start_load_kv: copy the store's span to its new position
            n = decision.hi - decision.lo
            src = self.store.read(session, "L0", decision.src_start, n)
            assert src is not None, "covers() promised a span that read() will not serve"
            self._write(blocks, decision.lo, decision.hi, [int(v) for v in src[:, 0, 0].tolist()])
            num_computed = decision.hi

        # vLLM prefills whatever is left
        self._write(blocks, num_computed, len(ids), ids[num_computed:])

        # every position must now hold its own token, or the model saw the wrong KV
        for pos, got in enumerate(self._read(blocks, 0, len(ids))):
            if got != ids[pos]:
                self.corruptions.append((self.turn, pos, ids[pos], got))

        # one scheduler step for the prefill, then one per generated token
        self._step(params, blocks, num_computed, len(ids), session)
        for k in range(max_tokens):
            cur = len(ids) + k
            if (cur // BLOCK) >= len(blocks):
                blocks.append(self._new_block())
            self._write(blocks, cur, cur + 1, [1])
            self._step(params, blocks, cur, cur + 1, session)

        self._publish(ids, blocks)
        return f"ok {self.turn}"

    def _step(self, params: dict, blocks, lo: int, hi: int, session) -> None:
        """One scheduler step's save, mirroring the connector's ``_plan_save``."""
        window = plan_save(self.store, session, params.get("save"), lo, hi)
        if window is None:
            if params.get("save") and hi > lo:
                self.refused_saves += 1
            return
        if params.get("save") == "full":
            params["save"] = True
        lo, hi = window
        values = self._read(blocks, lo, hi)
        self.store.write(
            session, "L0", lo, torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
        )
        self.saved_token_steps += hi - lo


def run_paged_session(
    turns: int, engine: FingerprintEngine, keep_last: int = 4, max_tokens: int = 3
):
    """Demote the oldest live message every turn: a front-of-view shrink edit, always."""
    server = MarathonServer(engine=engine, tokenizer=FakeTokenizer(), max_tokens=max_tokens)
    c = mclient.Client(mclient.local(server))
    rows = []
    for t in range(turns):
        engine.turn = t
        if t >= keep_last:
            # the cold tier's demotion: the oldest live message becomes a short stub
            d = t - keep_last
            c.edit("s", 2 * d, f"[cold #{d} {d:08x}]")
        rows.append(c.turn("s", f"Turn {t}. {FILLER * 3}"))
    return rows, server, c


def test_thirty_front_edits_never_corrupt_a_position():
    """The headline: 30 consecutive demote-style edits, every position verified."""
    engine = FingerprintEngine()
    run_paged_session(30, engine)

    assert engine.corruptions == [], (
        f"{len(engine.corruptions)} positions held the wrong token; "
        f"first three (turn, pos, want, got): {engine.corruptions[:3]}"
    )
    assert engine.declines == [], f"loads were declined: {engine.declines[:3]}"
    assert engine.refused_saves == 0


def test_front_edits_actually_reuse():
    """A run that never reuses anything would pass the test above trivially."""
    engine = FingerprintEngine()
    rows, _, _ = run_paged_session(30, engine)
    edits = [r for r in rows if r["reused_tokens"] > 0]
    # the staleness ceiling makes consecutive edit turns alternate reuse / recompute,
    # so roughly half of them reuse -- but it must never be none
    assert len(edits) >= 10, f"only {len(edits)} of 30 turns reused anything"
    assert all(r["phases"] == 2 for r in edits)


def test_front_edit_delta_is_negative():
    """A demotion shrinks the view, so the tail moves *earlier* — the untested sign."""
    engine = FingerprintEngine()
    server = MarathonServer(engine=engine, tokenizer=FakeTokenizer(), max_tokens=3)
    c = mclient.Client(mclient.local(server))
    for t in range(6):
        engine.turn = t
        c.turn("s", f"Turn {t}. {FILLER * 3}")

    c.edit("s", 0, "[cold #0 00000000]")
    engine.turn = 6
    r = c.turn("s", "Turn 6. " + FILLER * 3)
    assert r["deltas"], "the shrink edit produced no reusable segment at all"
    assert min(r["deltas"]) < 0, f"expected a negative delta, got {r['deltas']}"
    assert engine.corruptions == []


def test_full_save_is_one_shot_not_once_per_generated_token():
    """The paged cost bug: a latched 'full' re-gathers the prompt on every decode step."""
    engine = FingerprintEngine()
    server = MarathonServer(engine=engine, tokenizer=FakeTokenizer(), max_tokens=64)
    c = mclient.Client(mclient.local(server))
    for t in range(6):
        engine.turn = t
        c.turn("s", f"Turn {t}. {FILLER * 3}")
    before = engine.saved_token_steps
    c.edit("s", 0, "[cold #0 00000000]")
    engine.turn = 6
    r = c.turn("s", "Turn 6. " + FILLER * 3)
    cost = engine.saved_token_steps - before
    # one full rebuild plus one position per generated token, not 64 full rebuilds
    assert cost < 2 * r["prompt_tokens"], (
        f"the edit turn saved {cost} token-steps for a {r['prompt_tokens']}-token prompt; "
        "the 'full' save is latching across decode steps"
    )


def test_deep_session_store_stays_consistent_with_the_view():
    """After 30 front edits the store must still describe the *current* view."""
    engine = FingerprintEngine()
    run_paged_session(30, engine)
    assert engine.store.stats()["sessions"]["s"] > 0
    assert engine.store.misses == 0, "a covers() miss at depth means a silent recompute"


def test_staleness_budget_forces_a_periodic_honest_recompute():
    """Reused KV is stale by construction; without a ceiling the staleness ratchets.

    An edit turn re-saves the span it just loaded, so the reused vectors are never
    recomputed — in a paged session, where every turn edits the front of the view, turn
    N's reuse has accumulated N demotions' worth of drift against text that is no longer
    there. ``max_stale`` spends one honest recompute to reset that clock.
    """
    engine = FingerprintEngine()
    rows, server, _ = run_paged_session(30, engine)
    assert server.max_stale == 1

    refreshed = [i for i, r in enumerate(rows) if r["refreshed"]]
    assert refreshed, "no turn ever refreshed; the staleness clock is not running"
    # no run of reused turns may be longer than the budget
    assert all(r["stale"] <= server.max_stale for r in rows)
    gaps = [b - a for a, b in zip(refreshed, refreshed[1:], strict=False)]
    assert all(g <= server.max_stale + 1 for g in gaps), f"refresh gaps {gaps} exceed the budget"
    # and a refreshed turn is a real recompute: nothing handed to the connector
    assert all(rows[i]["reused_tokens"] == 0 for i in refreshed)


def test_staleness_budget_off_reuses_every_turn():
    """max_stale=0 disables reuse entirely; a large budget never refreshes."""
    engine = FingerprintEngine()
    server = MarathonServer(engine=engine, tokenizer=FakeTokenizer(), max_tokens=3, max_stale=10**6)
    c = mclient.Client(mclient.local(server))
    rows = []
    for t in range(12):
        engine.turn = t
        if t >= 4:
            c.edit("s", 2 * (t - 4), f"[cold #{t - 4} {t - 4:08x}]")
        rows.append(c.turn("s", f"Turn {t}. {FILLER * 3}"))
    assert not any(r["refreshed"] for r in rows)
    assert engine.corruptions == []
