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

from .shift_kernels import RopeShift, rope_shift, scatter_shifted, warmup
from .shift_store import DEFAULT_STORE_TOKENS, SessionTable, ShiftStore, slots

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# vLLM only configures handlers under the "vllm" logger namespace.
logger = init_logger("vllm.marathon_shift")

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
        self._store = ShiftStore(budget, device, allocate=(role == KVConnectorRole.WORKER))
        _STORES.append(self._store)

        # scheduler side
        self._table = SessionTable()
        self._params: dict[str, dict[str, Any]] = {}
        self._plans: dict[str, tuple[int, int, int]] = {}  # req -> (lo, hi, delta)
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

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        session = self._table.session_of(request.request_id)
        plan = (self._params.get(request.request_id) or {}).get("load")
        if not session or not plan:
            return 0, False
        dst_start, dst_end, delta = (
            int(plan["dst_start"]),
            int(plan["dst_end"]),
            int(plan["delta"]),
        )
        if num_computed_tokens < dst_start:
            # vLLM's own prefix hit stops before the reuse region starts; we cannot
            # ask it to compute a hole in the middle, so decline and let it recompute.
            logger.warning(
                "shift: local hit %d < dst_start %d, declining reuse",
                num_computed_tokens,
                dst_start,
            )
            return 0, False
        # whole blocks only, and always leave at least one token for vLLM to compute
        hi = min(dst_end, (request.num_prompt_tokens - 1) // self._bs * self._bs)
        hi -= hi % self._bs
        if hi <= num_computed_tokens:
            return 0, False
        if not self._store.covers(session, num_computed_tokens - delta, hi - num_computed_tokens):
            logger.warning(
                "shift: session %s no longer holds [%d,%d); declining reuse",
                session,
                num_computed_tokens - delta,
                hi - delta,
            )
            return 0, False
        self._plans[request.request_id] = (num_computed_tokens, hi, delta)
        return hi - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._need_load.add(request.request_id)

    def _plan_save(self, meta: ShiftConnectorMetadata, rid: str, lo: int, hi: int) -> None:
        """Record a save of the positions this step computed, if the store takes them.

        ``save="full"`` widens the save to ``[0, hi)`` instead of the positions this
        step actually computed. That is what an *edit* turn needs: the store is a flat
        position-indexed buffer, so a reused span cannot keep its old index once the
        edited span before it changed length — the new sequence and the old one no
        longer agree on where anything lives. Re-gathering the whole prompt out of the
        paged cache (where the loaded span, the prefix-cache hits and the freshly
        computed tokens are all resident by now) puts the store back into the *new*
        coordinates, which makes the session append-only again and the next edit an
        ordinary one. Loads for a step are all issued in ``start_load_kv`` before any
        ``save_kv_layer`` runs, so this re-read can never race the load that fed it.
        """
        session = self._table.session_of(rid)
        blocks = self._blocks.get(rid)
        if hi <= lo or not session or not blocks:
            return
        mode = (self._params.get(rid) or {}).get("save")
        if not mode:
            return
        if mode == "full":
            lo = 0
        if not self._store.reserve(session, lo, hi - lo):
            logger.warning("shift: session %s refused save of [%d,%d)", session, lo, hi)
            return
        meta.saves.append(_Save(session, slots(blocks, lo, hi, self._bs), lo))

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = ShiftConnectorMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            rid = req.req_id
            self._blocks[rid] = list(req.block_ids[0])
            if rid in self._need_load:
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
                scatter_shifted(
                    src.to(kv.device, non_blocking=True), kv, slot, self._bs, self._hnd, shift
                )
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
            slot = save.slots.to(kv_layer.device, non_blocking=True)
            self._store.write(
                save.session, layer_name, save.dst_start, kv_layer[self._paged(kv_layer, slot)]
            )

    def wait_for_save(self):
        return
