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

This connector does exactly that, for one session and one writer:

* SAVE  — every turn, the KV of the tokens vLLM actually computed is gathered out of
  the paged cache into a flat ``[max_tokens, num_kv_heads, 2*head_size]`` buffer per
  layer, indexed by absolute position. Because history is append-only until the edit,
  the buffer holds the previous turn's full KV by the time the edit arrives.
* LOAD  — the request carries a reuse plan in ``kv_transfer_params``:
  ``{"load": {"dst_start", "dst_end", "delta"}}``. The scheduler side reports
  ``dst_end - num_computed_tokens`` as externally available, so vLLM skips prefilling
  them; the worker side copies them in from the buffer, K re-rotated by ``delta``.

Caveats, honestly: single request in flight, no eviction, no multi-GPU, and the copy
is only done for whole blocks (vLLM counts matched tokens in block granularity), so a
ragged tail of ``S`` is left for vLLM to recompute. Registered lazily via
``KVTransferConfig(kv_connector="MarathonShiftConnector",
kv_connector_module_path="marathon.vllm_shift_connector")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# vLLM only configures handlers under the "vllm" logger namespace.
logger = init_logger("vllm.marathon_shift")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@dataclass
class _Load:
    slots: torch.Tensor  # [T] flat slot indices for the destination positions
    src_start: int  # first source position in the store
    delta: int


@dataclass
class _Save:
    slots: torch.Tensor  # [T] flat slot indices of the tokens computed this step
    dst_start: int  # first store position


@dataclass
class ShiftConnectorMetadata(KVConnectorMetadata):
    loads: list[_Load] = field(default_factory=list)
    saves: list[_Save] = field(default_factory=list)


class MarathonShiftConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._bs = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._max_store = int(extra.get("store_tokens", 16384))
        self._store_device = str(extra.get("store_device", "cuda"))

        # scheduler side
        self._params: dict[str, dict[str, Any]] = {}
        self._plans: dict[str, tuple[int, int, int]] = {}  # req -> (lo, hi, delta)
        self._need_load: set[str] = set()
        self._blocks: dict[str, list[int]] = {}

        # worker side
        self._kv: dict[str, torch.Tensor] = {}
        self._store: dict[str, torch.Tensor] = {}
        self._hnd: bool | None = None
        self._head_size: int | None = None
        self._rope: tuple[int, torch.Tensor, torch.Tensor] | None = None
        hf = vllm_config.model_config.hf_text_config
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        rotary = int(head_dim * getattr(hf, "partial_rotary_factor", 1.0))
        theta = float(getattr(hf, "rope_theta", 10000.0))
        self._inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
        )
        self._rotary = rotary

    # ------------------------------------------------------------------ helpers

    def _slots(self, block_ids: list[int], lo: int, hi: int) -> torch.Tensor:
        pos = torch.arange(lo, hi, dtype=torch.int64)
        return torch.as_tensor(block_ids, dtype=torch.int64)[pos // self._bs] * self._bs + (
            pos % self._bs
        )

    def _cos_sin(self, delta: int, device, dtype):
        if self._rope is None or self._rope[0] != delta:
            ang = float(delta) * self._inv_freq.to(device)
            emb = torch.cat((ang, ang), dim=-1)
            self._rope = (delta, emb.cos().to(dtype), emb.sin().to(dtype))
        return self._rope[1], self._rope[2]

    # ------------------------------------------------------------- scheduler side

    def on_new_request(self, request: "Request") -> None:
        self._params[request.request_id] = request.kv_transfer_params or {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        plan = (self._params.get(request.request_id) or {}).get("load")
        if not plan:
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
        self._plans[request.request_id] = (num_computed_tokens, hi, delta)
        return hi - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._need_load.add(request.request_id)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput"):
        meta = ShiftConnectorMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            rid = req.req_id
            self._blocks[rid] = list(req.block_ids[0])
            if rid in self._need_load:
                lo, hi, delta = self._plans[rid]
                meta.loads.append(
                    _Load(self._slots(self._blocks[rid], lo, hi), lo - delta, delta)
                )
                self._need_load.discard(rid)
            if (self._params.get(rid) or {}).get("save"):
                lo = req.num_computed_tokens
                hi = lo + scheduler_output.num_scheduled_tokens[rid]
                meta.saves.append(_Save(self._slots(self._blocks[rid], lo, hi), lo))

        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                self._blocks.setdefault(rid, []).extend(new_blocks[0])
            if not (self._params.get(rid) or {}).get("save"):
                continue
            lo = cached.num_computed_tokens[i]
            hi = lo + scheduler_output.num_scheduled_tokens[rid]
            if hi > lo:
                meta.saves.append(_Save(self._slots(self._blocks[rid], lo, hi), lo))

        for rid in scheduler_output.finished_req_ids:
            self._blocks.pop(rid, None)
            self._params.pop(rid, None)
            self._plans.pop(rid, None)
        return meta

    def request_finished(self, request: "Request", block_ids: list[int]):
        self._blocks.pop(request.request_id, None)
        self._params.pop(request.request_id, None)
        self._plans.pop(request.request_id, None)
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
        logger.info(
            "marathon shift connector: %d layers, kv shape %s, layout %s, store %d tok",
            len(kv_caches),
            tuple(shape),
            "HND" if self._hnd else "NHD",
            self._max_store,
        )

    def _paged(self, kv: torch.Tensor, slots: torch.Tensor) -> tuple:
        blk, off = slots // self._bs, slots % self._bs
        return (blk, slice(None), off) if self._hnd else (blk, off)

    def _ensure_store(self, name: str, kv: torch.Tensor) -> torch.Tensor:
        st = self._store.get(name)
        if st is None:
            heads = kv.shape[1] if self._hnd else kv.shape[2]
            st = torch.zeros(
                (self._max_store, heads, kv.shape[3]), dtype=kv.dtype, device=self._store_device
            )
            self._store[name] = st
        return st

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
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
        d = self._head_size
        for load in meta.loads:
            n = load.slots.numel()
            for name, kv in self._kv.items():
                st = self._store.get(name)
                if st is None:
                    logger.warning("shift: no stored KV for %s; skipping load", name)
                    return
                src = st[load.src_start : load.src_start + n].to(kv.device, non_blocking=True)
                k = src[..., :d]
                if load.delta:
                    cos, sin = self._cos_sin(load.delta, kv.device, torch.float32)
                    kf = k.to(torch.float32)
                    k = (kf * cos + _rotate_half(kf) * sin).to(kv.dtype)
                slots = load.slots.to(kv.device, non_blocking=True)
                kv[self._paged(kv, slots)] = torch.cat((k, src[..., d:]), dim=-1)
            logger.info(
                "shift: loaded %d tokens x %d layers from store[%d:%d], delta=%d",
                n, len(self._kv), load.src_start, load.src_start + n, load.delta,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kw):
        meta = self._get_connector_metadata()
        if not isinstance(meta, ShiftConnectorMetadata) or not meta.saves:
            return
        if self._hnd is None:
            self.register_kv_caches({layer_name: kv_layer})
            self._kv = {}
        st = self._ensure_store(layer_name, kv_layer)
        for save in meta.saves:
            n = save.slots.numel()
            if save.dst_start + n > self._max_store:
                continue
            slots = save.slots.to(kv_layer.device, non_blocking=True)
            st[save.dst_start : save.dst_start + n] = kv_layer[
                self._paged(kv_layer, slots)
            ].to(st.device, non_blocking=True)

    def wait_for_save(self):
        return
