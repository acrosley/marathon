"""Position-shifted KV reuse inside vLLM: a KVConnector that re-rotates cached keys.

The serving-side half of the result in ``kvshift.py``. When a mid-history edit turns

    old sequence   [ P ][ E  ][ S ][ ... ]
    new sequence   [ P ][ E' ][ S ][ new ]

vLLM's prefix cache can only reuse ``P``; everything from the edit onwards is
recomputed. Nothing about ``S`` changed except *where it sits*: its values carry no
position at all, and its keys carry RoPE, which is a rotation — so a key computed at
position ``p`` is moved to ``p + d`` exactly by rotating it once more by ``d *
inv_freq`` (:func:`marathon.kvshift.rerotate_keys`, unit-tested in
``tests/test_kvshift.py``). So ``S`` can be *copied* into its new block slots with the
K half re-rotated, and vLLM only has to prefill ``E'`` and the genuinely new tokens.

Every request carries its reuse plan in ``kv_transfer_params``::

    {"session": "<id>", "load": {"dst_start", "dst_end", "delta"}, "save": true | "full"}

* **session** — the store is keyed by it (:class:`marathon.shift_store.ShiftStore`),
  so sessions never read each other's KV, and a request without one is pass-through:
  no load, no save, vLLM behaves exactly as it would with no connector. v1 allows one
  in-flight writer per session (:class:`marathon.shift_store.SessionTable`); a second
  concurrent request on the same session is logged and served with no reuse at all,
  never with a wrong load — which also means a load can never overlap an in-flight
  save on the same session.
* **save** — after each layer is written, the KV of the positions this request
  actually computed (``num_computed_tokens`` onward, per scheduler step, so chunked
  prefill and continued prefills are covered) is gathered out of the paged cache into
  the session's flat ``[capacity, num_kv_heads, 2*head_size]`` buffer, indexed by
  absolute position. Positions are append-only within a turn; a save at an earlier
  ``dst_start`` truncates everything above it, which is exactly what an edit means.
  ``save="full"`` re-gathers the *whole* prompt rather than only what this step
  computed, which is how an edit turn puts the store back into the new position
  coordinates; without it a session's second edit finds a store indexed by the layout
  the first edit replaced, and degrades to a recompute.
* **load** — the scheduler side reports ``dst_end - num_computed_tokens`` as
  externally available, so vLLM skips prefilling them; the worker copies them in from
  the session's buffer with K re-rotated by ``delta``. The scheduler side runs the
  same store bookkeeping as the worker (without tensors) and declines a load whose
  source positions have been evicted or truncated, so a miss is a recompute rather
  than garbage.

Memory: the total store budget across sessions is ``MARATHON_STORE_TOKENS`` tokens
(default 32768) or ``kv_connector_extra_config["store_tokens"]``; see
:mod:`marathon.shift_store` for the per-token cost (164 KB/token on Qwen3-14B).
Eviction is LRU across whole sessions. :func:`stats` returns the merged counters.

Known-unsafe / untested: tensor parallelism (each worker would keep its own store;
never tried with TP>1), preemption and recompute of a request mid-flight, and chunked
prefill *interleaved with* a load on the same request. Whole-block granularity is
still a constraint — vLLM counts externally matched tokens in whole blocks, so a
ragged head/tail of a reused run is left for vLLM to recompute. Registered lazily via
``KVTransferConfig(kv_connector="MarathonShiftConnector",
kv_connector_module_path="marathon.vllm_shift_connector")``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from . import gapfill_channel
from .shift_kernels import RopeShift, rope_shift, scatter_shifted, warmup
from .shift_store import (
    DEFAULT_STORE_TOKENS,
    SessionTable,
    ShiftStore,
    plan_load,
    plan_save,
    slots,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# vLLM only configures handlers under the "vllm" logger namespace.
logger = init_logger("vllm.marathon_shift")

#: ``MARATHON_VERIFY_LOAD=1`` reads every loaded span back out of the paged cache and
#: compares it against the torch reference (:func:`marathon.kvshift.rerotate_keys` on the
#: same source rows). The fingerprint harness in ``tests/test_paged_depth.py`` models KV
#: as token ids, so it proves *placement* and can never catch a wrong rotation angle, a
#: write into the wrong layer, or a layout/stride mistake. This catches all three, in the
#: live serving path, at the cost of a read-back per layer -- diagnostic only, off by
#: default.
VERIFY_LOAD = os.environ.get("MARATHON_VERIFY_LOAD", "") not in ("", "0")

#: ``MARATHON_GAPFILL=1`` switches to the single-request path: a turn sends *all* its
#: reused segments as ``kv_transfer_params["loads"]``, the connector declares a non-prefix
#: match through :mod:`marathon.gapfill_channel`, and the engine prefills only the gaps.
#: Inert unless ``scripts/patch_vllm_gapfill.py`` has been applied -- without the patch
#: the scheduler ignores the channel and the request would be told nothing is matched, so
#: the connector declines instead of guessing.
GAPFILL = os.environ.get("MARATHON_GAPFILL", "") not in ("", "0")

# The scheduler-side and worker-side connectors are separate instances (and, with a
# non-uniprocess executor, separate processes). Both register here so `stats()` can
# report whatever is reachable from the caller's process.
_STORES: list[ShiftStore] = []

_COUNTERS = ("hits", "misses", "evictions", "refusals", "loaded_token_layers", "saved_token_layers")


def stats() -> dict:
    """Merged store counters: tokens per session, hits, loads, saves, evictions."""
    out: dict[str, Any] = dict.fromkeys(_COUNTERS, 0)
    out.update(sessions={}, used_tokens=0, budget_tokens=0)
    for store in _STORES:
        s = store.stats()
        out["budget_tokens"] = s["budget_tokens"]
        out["used_tokens"] = max(out["used_tokens"], s["used_tokens"])
        for name, filled in s["sessions"].items():
            out["sessions"][name] = max(out["sessions"].get(name, 0), filled)
        for k in _COUNTERS:
            out[k] += s[k]
    return out


@dataclass
class _Load:
    session: str
    slots: torch.Tensor  # [T] flat slot indices for the destination positions
    src_start: int  # first source position in the session's store
    delta: int


@dataclass
class _Save:
    session: str
    slots: torch.Tensor  # [T] flat slot indices of the tokens computed this step
    dst_start: int  # first store position
    #: (block, offset) index tensors on the compute device, built once per step and
    #: reused across all layers -- see ``MarathonShiftConnector._gather``
    idx: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class ShiftConnectorMetadata(KVConnectorMetadata):
    loads: list[_Load] = field(default_factory=list)
    saves: list[_Save] = field(default_factory=list)


class MarathonShiftConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._bs = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        budget = int(
            extra.get("store_tokens", os.environ.get("MARATHON_STORE_TOKENS", DEFAULT_STORE_TOKENS))
        )
        device = str(extra.get("store_device", "cuda"))
        # positions a single session will ever need, when the caller knows: allocated
        # once, so no save has to grow a buffer on a GPU that has no headroom left
        cap = int(extra.get("session_tokens", 0) or 0)
        self._store = ShiftStore(
            budget, device, allocate=(role == KVConnectorRole.WORKER), session_cap=cap
        )
        _STORES.append(self._store)

        # scheduler side
        self._table = SessionTable()
        self._params: dict[str, dict[str, Any]] = {}
        self._plans: dict[str, tuple[int, int, int]] = {}  # req -> (lo, hi, delta)
        self._multi: dict[str, list[dict]] = {}  # req -> every segment, single-request mode
        self._need_load: set[str] = set()
        self._blocks: dict[str, list[int]] = {}

        # worker side
        self._kv: dict[str, torch.Tensor] = {}
        self._hnd: bool | None = None
        self._head_size: int | None = None
        self._rope: RopeShift | None = None
        hf = vllm_config.model_config.hf_text_config
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        rotary = int(head_dim * getattr(hf, "partial_rotary_factor", 1.0))
        theta = float(getattr(hf, "rope_theta", 10000.0))
        self._inv_freq = 1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary))
        self._rotary = rotary

    # ------------------------------------------------------------------ helpers

    def _shift(self, delta: int, device) -> RopeShift:
        """Rotation tables for δ, rebuilt only when δ changes (once per load)."""
        if self._rope is None or self._rope.delta != delta or self._rope.cos.device != device:
            self._rope = rope_shift(delta, self._head_size, self._inv_freq, device)
        return self._rope

    # ------------------------------------------------------------- scheduler side

    def on_new_request(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        self._params[request.request_id] = params
        session = params.get("session")
        if not session:
            return  # no session id: pass-through, exactly as if there were no connector
        if not self._table.acquire(request.request_id, str(session)):
            logger.warning(
                "shift: session %s already has an in-flight writer; request %s runs "
                "with no reuse (single writer per session is the v1 rule)",
                session,
                request.request_id,
            )

    def _accept_multi(self, request: Request, session: str, loads: list, num_computed: int):
        """Block-align and coverage-check every segment of a single-request load.

        Returns the segments the store can actually serve. A segment is dropped -- never
        guessed at -- if alignment leaves it under a block, if it lies inside what the
        engine already has locally, or if the store no longer holds its source.
        """
        limit = (request.num_prompt_tokens - 1) // self._bs * self._bs
        floor = -(-num_computed // self._bs) * self._bs
        out = []
        for ld in loads:
            delta = int(ld["delta"])
            lo = max(-(-int(ld["dst_start"]) // self._bs) * self._bs, floor)
            hi = min(int(ld["dst_end"]), limit) // self._bs * self._bs
            if hi - lo < self._bs:
                continue
            if not self._store.covers(session, lo - delta, hi - lo):
                logger.warning("shift: gapfill dropping [%d,%d): store lost the source", lo, hi)
                continue
            out.append({"dst_start": lo, "dst_end": hi, "delta": delta})
        return out

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        session = self._table.session_of(request.request_id)
        params = self._params.get(request.request_id) or {}

        multi = params.get("loads")
        if GAPFILL and session and multi:
            from .gapfill import plan_gaps

            accepted = self._accept_multi(request, session, multi, num_computed_tokens)
            if not accepted:
                return 0, False
            gp = plan_gaps(
                accepted, self._bs, request.num_prompt_tokens, local_hit=num_computed_tokens
            )
            gapfill_channel.offer(request.request_id, gp.compute, gp.filled_tokens)
            self._multi[request.request_id] = accepted
            logger.info(
                "shift: gapfill %d segments, %d/%d tokens filled, %d to compute",
                len(accepted),
                gp.filled_tokens,
                gp.n_prompt,
                len(gp.compute),
            )
            return max(gp.filled_tokens - num_computed_tokens, 0), False

        req = params.get("load")
        decision, why = plan_load(
            self._store,
            session,
            req,
            num_computed_tokens,
            request.num_prompt_tokens,
            self._bs,
        )
        if decision is None:
            if session and req:
                logger.warning("shift: session %s declining reuse: %s", session, why)
            return 0, False
        self._plans[request.request_id] = (decision.lo, decision.hi, decision.delta)
        return decision.hi - decision.lo, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._need_load.add(request.request_id)

    def _plan_save(self, meta: ShiftConnectorMetadata, rid: str, lo: int, hi: int) -> None:
        """Record a save of the positions this step computed, if the store takes them.

        ``save="full"`` widens the save to ``[0, hi)``. That is what an *edit* turn
        needs: the store is a flat position-indexed buffer, so a reused span cannot keep
        its old index once the edited span before it changed length — the new sequence
        and the old one no longer agree on where anything lives. Re-gathering the whole
        prompt out of the paged cache (where the loaded span, the prefix-cache hits and
        the freshly computed tokens are all resident by now) puts the store back into
        the *new* coordinates, which makes the session append-only again and the next
        edit an ordinary one. Loads for a step are all issued in ``start_load_kv``
        before any ``save_kv_layer`` runs, so this re-read cannot race the load that fed
        it.

        It is a *one-shot* widening, downgraded to an ordinary incremental save as soon
        as it has been planned once. ``_plan_save`` runs on every scheduler step,
        decode steps included, so leaving it latched would re-gather the entire prompt
        across every layer once per generated token — measured 2026-08-19 as the
        dominant cost of a paged session, where every turn is an edit turn and every
        turn generates a real answer.
        """
        session = self._table.session_of(rid)
        blocks = self._blocks.get(rid)
        if not blocks:
            return
        params = self._params.get(rid) or {}
        window = plan_save(self._store, session, params.get("save"), lo, hi)
        if window is None:
            if session and params.get("save") and hi > lo:
                logger.warning("shift: session %s refused save of [%d,%d)", session, lo, hi)
            return
        if params.get("save") == "full":
            params["save"] = True
        lo, hi = window
        meta.saves.append(_Save(session or "", slots(blocks, lo, hi, self._bs), lo))

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = ShiftConnectorMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            rid = req.req_id
            self._blocks[rid] = list(req.block_ids[0])
            if rid in self._need_load and rid in self._multi:
                for seg in self._multi[rid]:
                    meta.loads.append(
                        _Load(
                            self._table.session_of(rid) or "",
                            slots(self._blocks[rid], seg["dst_start"], seg["dst_end"], self._bs),
                            seg["dst_start"] - seg["delta"],
                            seg["delta"],
                        )
                    )
                self._need_load.discard(rid)
            elif rid in self._need_load:
                lo, hi, delta = self._plans[rid]
                meta.loads.append(
                    _Load(
                        self._table.session_of(rid) or "",
                        slots(self._blocks[rid], lo, hi, self._bs),
                        lo - delta,
                        delta,
                    )
                )
                self._need_load.discard(rid)
            lo = req.num_computed_tokens
            self._plan_save(meta, rid, lo, lo + scheduler_output.num_scheduled_tokens[rid])

        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                self._blocks.setdefault(rid, []).extend(new_blocks[0])
            lo = cached.num_computed_tokens[i]
            self._plan_save(meta, rid, lo, lo + scheduler_output.num_scheduled_tokens[rid])

        for rid in scheduler_output.finished_req_ids:
            self._forget(rid)
        return meta

    def _forget(self, rid: str) -> None:
        gapfill_channel.release(rid)
        self._multi.pop(rid, None)
        self._blocks.pop(rid, None)
        self._params.pop(rid, None)
        self._plans.pop(rid, None)
        self._need_load.discard(rid)
        self._table.release(rid)

    def request_finished(self, request: Request, block_ids: list[int]):
        self._forget(request.request_id)
        # The engine core is a separate process from whoever called generate(), so
        # `stats()` is only reachable in-process; logging it is how a probe run gets
        # the numbers back. One line per finished request is cheap at probe rates.
        logger.info("shift: store %s", self._store.stats())
        return False, None

    # ---------------------------------------------------------------- worker side

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv = kv_caches
        shape = next(iter(kv_caches.values())).shape
        assert len(shape) == 4, f"expected fused 4-D KV layout, got {tuple(shape)}"
        # [num_blocks, num_heads, block_size, 2*head_size] (HND) or
        # [num_blocks, block_size, num_heads, 2*head_size] (NHD)
        self._hnd = shape[2] == self._bs and shape[1] != self._bs
        if shape[1] == self._bs and shape[2] == self._bs:
            self._hnd = False  # ambiguous; NHD is vLLM's default
        self._head_size = shape[3] // 2
        one = next(iter(kv_caches.values()))
        if one.is_cuda:
            warmup(one, self._bs, self._hnd)
        logger.info(
            "marathon shift connector: %d layers, kv shape %s, layout %s, store %d tok",
            len(kv_caches),
            tuple(shape),
            "HND" if self._hnd else "NHD",
            self._store.budget,
        )

    def _paged(self, kv: torch.Tensor, slot: torch.Tensor) -> tuple:
        blk, off = slot // self._bs, slot % self._bs
        return (blk, slice(None), off) if self._hnd else (blk, off)

    def _gather(self, kv: torch.Tensor, save: _Save) -> torch.Tensor:
        """The tokens of ``save`` out of the paged cache, as ``[T, heads, 2*head_size]``.

        Two things matter here and neither is obvious. First, the slot transfer and the
        block/offset split are per *step*, not per layer: doing them inside the 40-layer
        loop repeats an H2D copy and two integer divisions over the whole span forty
        times. Second, on the HND layout the natural index ``kv[blk, :, off]`` separates
        two advanced indices with a slice, which is PyTorch's slow gather path; a
        ``permute`` to NHD order first makes them adjacent and hits the fast one. The
        permute is a view, so it costs nothing.

        Honest note on why this exists: it was written to explain the 5.2 s refresh turns
        measured 2026-08-20 on Qwen3-14B, and it did not — the same run after this change
        came back at 6.0 s, i.e. unchanged within run-to-run noise. So the refresh cost
        lives somewhere else and is still open. This is kept only because it is strictly
        less work than what it replaced and is pinned value-identical by
        ``test_hnd_gather_permute_matches_separated_advanced_indexing``; it is not a fix
        for anything measured.
        """
        if save.idx is None:
            slot = save.slots.to(kv.device, non_blocking=True)
            save.idx = (slot // self._bs, slot % self._bs)
        blk, off = save.idx
        src = kv.permute(0, 2, 1, 3) if self._hnd else kv
        return src[blk, off]

    def _verify(self, layer: str, kv, row, slot, delta: int) -> None:
        """Read a just-scattered span back and compare it to the torch reference.

        ``row`` is what the store handed us, ``[n, heads, 2*head_size]`` with K in the
        first half. The reference re-rotates K with :func:`marathon.kvshift.rerotate_keys`
        -- the same function ``kvshift`` is unit-tested against -- and leaves V alone,
        then checks that is what actually landed in the paged cache at ``slot``. A
        nonzero difference localises the bug to this write: wrong angle, wrong layer,
        wrong slot, or wrong layout.
        """
        from .kvshift import rerotate_keys

        d = row.shape[-1] // 2
        if 2 * int(self._inv_freq.numel()) != d:
            logger.warning(
                "shift-verify: partial rotary (%d of %d) unsupported; skipped",
                2 * int(self._inv_freq.numel()),
                d,
            )
            return
        want_k = rerotate_keys(
            row[..., :d].to(torch.float32), int(delta), self._inv_freq.to(row.device)
        )
        want = torch.cat((want_k, row[..., d:].to(torch.float32)), dim=-1)
        got = kv[self._paged(kv, slot)].to(torch.float32)
        diff = (got - want).abs()
        rel = diff.max() / want.abs().max().clamp_min(1e-6)
        logger.info(
            "shift-verify: layer=%s delta=%d n=%d max_abs=%.4g max_rel=%.4g mean_abs=%.4g",
            layer,
            delta,
            row.shape[0],
            float(diff.max()),
            float(rel),
            float(diff.mean()),
        )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, ShiftConnectorMetadata) or not meta.loads:
            return
        if not self._kv:
            self._kv = {
                n: layer.kv_cache[forward_context.virtual_engine]
                if isinstance(getattr(layer, "kv_cache", None), list)
                else layer.kv_cache
                for n, layer in forward_context.no_compile_layers.items()
                if getattr(layer, "kv_cache", None) is not None
            }
            self.register_kv_caches(self._kv)
        device = next(iter(self._kv.values())).device
        for load in meta.loads:
            n = load.slots.numel()
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
            _bytes = 0
            # δ and the destination slots are the same for every layer, so the rotation
            # tables and the slot transfer are hoisted out of the per-layer loop; each
            # layer is then one fused read-rotate-scatter pass (see marathon.shift_kernels).
            shift = self._shift(load.delta, device) if load.delta else None
            slot = load.slots.to(device, non_blocking=True)
            for name, kv in self._kv.items():
                src = self._store.read(load.session, name, load.src_start, n)
                if src is None:
                    logger.error(
                        "shift: session %s has no stored KV for %s; load skipped",
                        load.session,
                        name,
                    )
                    return
                row = src.to(kv.device, non_blocking=True)
                scatter_shifted(row, kv, slot, self._bs, self._hnd, shift)
                if VERIFY_LOAD:
                    self._verify(name, kv, row, slot, load.delta)
                _bytes += src.numel() * src.element_size()
            torch.cuda.synchronize()
            _ms = (time.perf_counter() - _t0) * 1e3
            logger.info(
                "shift: session %s loaded %d tokens x %d layers from store[%d:%d], "
                "delta=%d, copy_ms=%.2f (%.2f MB, %.1f GB/s)",
                load.session,
                n,
                len(self._kv),
                load.src_start,
                load.src_start + n,
                load.delta,
                _ms,
                _bytes / 2**20,
                _bytes / 2**30 / max(_ms / 1e3, 1e-9),
            )
            logger.info("shift: store %s", self._store.stats())

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kw):
        meta = self._get_connector_metadata()
        if not isinstance(meta, ShiftConnectorMetadata) or not meta.saves:
            return
        if self._hnd is None:
            self.register_kv_caches({layer_name: kv_layer})
            self._kv = {}
        for save in meta.saves:
            n = save.slots.numel()
            # Mirrors the scheduler's reserve for this step; idempotent across layers.
            if not self._store.reserve(save.session, save.dst_start, n):
                continue
            self._store.write(
                save.session, layer_name, save.dst_start, self._gather(kv_layer, save)
            )

    def wait_for_save(self):
        return
