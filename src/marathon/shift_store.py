"""Session-keyed KV store for the shift connector: budget, LRU, single-writer guard.

The parts of :mod:`marathon.vllm_shift_connector` that are pure bookkeeping, split
out so they can be unit-tested on CPU without vLLM. Two objects live here:

:class:`SessionTable`
    Who owns a session right now. v1's concurrency rule is one writer per session
    (DESIGN.md, protocol.md), so if two in-flight requests claim the same session id
    the second is refused and served as a plain no-reuse request. A refusal costs a
    recompute; a wrong load costs a wrong answer.

:class:`ShiftStore`
    Per session, a position-indexed buffer of that session's KV, one tensor per
    layer, shaped ``[capacity, num_kv_heads, 2 * head_size]`` — index is the token's
    absolute position in the prompt, so a reuse plan's ``src_start`` is a slice.
    Buffers double from a small floor; when the total across sessions would exceed the
    budget, whole sessions are evicted least-recently-used first.

**Per-token cost.** One position costs ``num_layers * num_kv_heads * head_size * 2 *
dtype_bytes``. On Qwen3-14B (40 layers, 8 KV heads, head_size 128, bf16) that is
**164 KB/token**: a 32768-token budget is 5.4 GB of GPU memory, which is why
``gpu_memory_utilization`` has to leave room for it. On Qwen3-0.6B (28 layers, 8
heads, 128) it is 115 KB/token. The budget is counted in *tokens*, summed over
sessions, and is set by ``MARATHON_STORE_TOKENS`` (default 32768) or the connector's
``kv_connector_extra_config["store_tokens"]``.

The store is *mirrored*: the scheduler-side connector instance runs the same
bookkeeping with ``allocate=False`` (no tensors) so it can answer "do I actually
still hold these positions?" before promising vLLM that a span needs no prefill.
Both sides see the same sequence of saves in the same order, so their views agree;
without that, an eviction on the worker would turn into silently garbage KV.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

# A session's buffers double geometrically from a small floor, so a session pays
# O(log n) reallocations over its whole life rather than one per turn. Measured
# 2026-08-19 on Qwen3-14B: growing a ~2 GB store in *fixed 1024-token steps* reallocates
# all 40 layer buffers every other turn, and the allocator churn slowed every turn of the
# run 2-4x while cutting the load copy from 57 GB/s to 4.7 GB/s. Doubling is what fixes
# that, not a large first allocation: allocating a fixed 16384-token slab up front made
# *one session* fill a 24576-token budget, so a second session evicted the first and both
# reallocated all 40 buffers on every single turn -- 2.7 GB allocated, zeroed and freed
# per turn on 14B, which is the memory-bound stall measured on the paged workload
# (findings 2026-08-19). The floor keeps short sessions from reallocating for a while;
# the doubling keeps long ones from reallocating often.
SLAB = 2048  # *floor* for a session's first allocation; capacity doubles from there
DEFAULT_STORE_TOKENS = 32768


def slots(block_ids: list[int], lo: int, hi: int, block_size: int) -> torch.Tensor:
    """Flat slot indices of positions ``[lo, hi)`` given a request's block table."""
    pos = torch.arange(lo, hi, dtype=torch.int64)
    table = torch.as_tensor(block_ids, dtype=torch.int64)
    return table[pos // block_size] * block_size + (pos % block_size)


class SessionTable:
    """Single-writer-per-session guard, keyed by in-flight request id."""

    def __init__(self) -> None:
        self._owner: dict[str, str] = {}  # session -> request id
        self._session: dict[str, str] = {}  # request id -> session
        self.conflicts = 0

    def acquire(self, request_id: str, session: str) -> bool:
        """Claim ``session`` for ``request_id``. False if another request holds it."""
        held = self._owner.get(session)
        if held is not None and held != request_id:
            self.conflicts += 1
            return False
        self._owner[session] = request_id
        self._session[request_id] = session
        return True

    def session_of(self, request_id: str) -> str | None:
        return self._session.get(request_id)

    def release(self, request_id: str) -> None:
        session = self._session.pop(request_id, None)
        if session is not None and self._owner.get(session) == request_id:
            del self._owner[session]

    def __len__(self) -> int:
        return len(self._owner)


@dataclass
class _Entry:
    capacity: int = 0  # positions the buffers can hold
    base: int = -1  # first position ever written; below it the store holds nothing
    filled: int = 0  # positions [base, filled) hold valid KV for the current history
    layers: dict[str, torch.Tensor] = field(default_factory=dict)


class ShiftStore:
    """Position-indexed KV per session, under a total token budget with LRU eviction."""

    def __init__(
        self,
        budget_tokens: int = DEFAULT_STORE_TOKENS,
        device: str = "cuda",
        allocate: bool = True,
        slab: int | None = None,
    ) -> None:
        self.budget = int(budget_tokens)
        self.slab = min(int(slab or SLAB), self.budget)
        self.device = device
        self.allocate = allocate
        self._sessions: OrderedDict[str, _Entry] = OrderedDict()  # LRU: oldest first
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.refusals = 0
        self.loaded_token_layers = 0
        self.saved_token_layers = 0

    # ----------------------------------------------------------------- internals

    def _touch(self, session: str) -> _Entry:
        entry = self._sessions.get(session)
        if entry is None:
            entry = _Entry()
            self._sessions[session] = entry
        else:
            self._sessions.move_to_end(session)
        return entry

    @property
    def used(self) -> int:
        return sum(e.capacity for e in self._sessions.values())

    def _grow(self, session: str, entry: _Entry, want: int) -> None:
        # geometric from the session's *current* capacity, floored at the slab size:
        # never allocate a session more than it has shown it needs, and never grow it
        # by a constant step either
        size = max(self.slab, entry.capacity * 2)
        while size < want:
            size *= 2
        want = min(self.budget, size)
        # Evict least-recently-used *other* sessions until the new capacity fits.
        while self.used - entry.capacity + want > self.budget:
            victim = next((s for s in self._sessions if s != session), None)
            if victim is None:
                break
            self.drop(victim)
            self.evictions += 1
        for name, old in list(entry.layers.items()):
            new = torch.zeros((want, *old.shape[1:]), dtype=old.dtype, device=old.device)
            new[: min(entry.filled, old.shape[0])] = old[: min(entry.filled, old.shape[0])]
            entry.layers[name] = new
        entry.capacity = want

    # -------------------------------------------------------------------- public

    def reserve(self, session: str, dst_start: int, n: int) -> bool:
        """Make room for positions ``[dst_start, dst_start + n)`` of ``session``.

        Returns False — and stores nothing — if the span would leave a hole after the
        positions already held, or does not fit in the budget even alone. Writing at a
        ``dst_start`` at or below ``filled`` is allowed and *truncates* anything above
        the span: that is what an edit does, since every position from the edit onward
        is recomputed and the old KV there is stale.

        Idempotent for a repeated ``(session, dst_start, n)``, so the worker may call
        it once per layer while the scheduler calls it once per step.
        """
        need = dst_start + n
        entry = self._touch(session)
        first = entry.base < 0
        # A request only computes what vLLM did not already have: its first save
        # therefore starts at its prefix-cache hit, not at position 0. That head is
        # served by vLLM's own prefix cache on every later turn, so the store never
        # needs it — but it must not claim to hold it either, hence `base`.
        if need > self.budget or n <= 0 or (not first and dst_start > entry.filled):
            self.refusals += 1
            return False
        if need > entry.capacity:
            self._grow(session, entry, need)
        if first:
            entry.base = dst_start
        entry.filled = need
        return True

    def covers(self, session: str, start: int, n: int) -> bool:
        """Whether ``[start, start + n)`` is currently held for ``session``."""
        entry = self._sessions.get(session)
        ok = (
            entry is not None
            and entry.base >= 0
            and start >= entry.base
            and n > 0
            and start + n <= entry.filled
        )
        if ok:
            self.hits += 1
            self._touch(session)
        else:
            self.misses += 1
        return ok

    def write(self, session: str, layer: str, dst_start: int, src: torch.Tensor) -> None:
        """Copy ``src`` (``[n, heads, 2*head_size]``) into the session's buffer."""
        entry = self._sessions.get(session)
        n = src.shape[0]
        if entry is None or dst_start + n > entry.capacity or not self.allocate:
            return
        buf = entry.layers.get(layer)
        if buf is None:
            buf = torch.zeros((entry.capacity, *src.shape[1:]), dtype=src.dtype, device=self.device)
            entry.layers[layer] = buf
        buf[dst_start : dst_start + n] = src.to(buf.device, non_blocking=True)
        self.saved_token_layers += n

    def read(self, session: str, layer: str, start: int, n: int) -> torch.Tensor | None:
        entry = self._sessions.get(session)
        if entry is None or start < entry.base or entry.base < 0 or start + n > entry.filled:
            return None
        buf = entry.layers.get(layer)
        if buf is None:
            return None
        self.loaded_token_layers += n
        # Touch for the same reason (and in the same order) as the scheduler mirror's
        # ``covers``: the two LRU orderings have to stay identical or they would pick
        # different eviction victims.
        self._touch(session)
        return buf[start : start + n]

    def drop(self, session: str) -> None:
        self._sessions.pop(session, None)

    def stats(self) -> dict:
        return {
            "budget_tokens": self.budget,
            "used_tokens": self.used,
            "sessions": {s: max(e.filled - max(e.base, 0), 0) for s, e in self._sessions.items()},
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "refusals": self.refusals,
            "loaded_token_layers": self.loaded_token_layers,
            "saved_token_layers": self.saved_token_layers,
        }


@dataclass(frozen=True)
class LoadDecision:
    """A resolved load: destination ``[lo, hi)``, source at ``lo - delta``."""

    lo: int
    hi: int
    delta: int

    @property
    def src_start(self) -> int:
        return self.lo - self.delta


def plan_load(
    store: ShiftStore,
    session: str | None,
    request: dict | None,
    num_computed: int,
    num_prompt_tokens: int,
    block_size: int,
) -> tuple[LoadDecision | None, str]:
    """Decide whether a request's ``load`` can be served. Returns ``(decision, why)``.

    Pure bookkeeping, factored out of the connector's ``get_num_new_matched_tokens`` so
    the same code can be driven on CPU without vLLM (``tests/test_paged_depth.py``).
    A ``None`` decision always means *recompute*, never *guess*: every rejection path
    here costs prefill and none of them can produce wrong KV.
    """
    if not session or not request:
        return None, "no session or no load requested"
    lo, dst_end, delta = (
        int(num_computed),
        int(request["dst_end"]),
        int(request["delta"]),
    )
    if num_computed < int(request["dst_start"]):
        # vLLM's own prefix hit stops before the reuse region starts; we cannot ask it
        # to compute a hole in the middle, so decline and let it recompute.
        return None, f"local hit {num_computed} < dst_start {request['dst_start']}"
    # whole blocks only, and always leave at least one token for vLLM to compute
    hi = min(dst_end, (num_prompt_tokens - 1) // block_size * block_size)
    hi -= hi % block_size
    if hi <= lo:
        return None, f"nothing left after block alignment (hi={hi} <= lo={lo})"
    if not store.covers(session, lo - delta, hi - lo):
        return None, f"store no longer holds [{lo - delta},{hi - delta})"
    return LoadDecision(lo, hi, delta), "ok"


def plan_save(
    store: ShiftStore, session: str | None, mode: object, lo: int, hi: int
) -> tuple[int, int] | None:
    """Decide the position range a step should save. Factored out of ``_plan_save``.

    ``mode="full"`` widens the save to ``[0, hi)``: an edit turn moves everything after
    the edit, so the store — a flat position-indexed buffer — has to be rewritten in the
    new coordinates or the next edit plans against a layout that no longer exists.
    """
    if hi <= lo or not session or not mode:
        return None
    if mode == "full":
        lo = 0
    if not store.reserve(session, lo, hi - lo):
        return None
    return lo, hi
