"""The hand-off between a connector that plans gaps and a vLLM patched to honour them.

``scripts/patch_vllm_gapfill.py`` injects two small edits into vLLM: the scheduler
``take``s a gap plan a connector has ``offer``ed for a request and republishes it, and the
model runner reads ``active()`` to overwrite that request's positions. Keeping the channel
here rather than inside the connector means the patch imports one stable module with a
four-function surface, so a vLLM upgrade only ever breaks the anchors, never the protocol.

Single process, single GPU: the v1 scheduler and worker share a process, which is what
makes a module-level dict a legitimate channel. Tensor parallelism would give each worker
its own copy and is not supported.

Every entry is keyed by request id and must be released when the request finishes;
:func:`release` is idempotent so the connector can call it from ``request_finished``
without tracking whether a plan was ever offered.
"""

from __future__ import annotations

import numpy as np

#: request id -> (positions the engine must compute, tokens the connector supplies)
_offered: dict[str, tuple[np.ndarray, int]] = {}
_active: dict[str, tuple[np.ndarray, int]] = {}


def offer(request_id: str, positions, matched: int) -> None:
    """Connector side: this request needs only ``positions`` computed."""
    arr = np.asarray(positions, dtype=np.int64)
    _offered[str(request_id)] = (arr, int(matched))


def take(request_id: str) -> tuple[np.ndarray, int] | None:
    """Scheduler side: consume an offer, if one was made for this request."""
    return _offered.pop(str(request_id), None)


def publish(request_id: str, positions, matched: int) -> None:
    """Scheduler side: make the plan visible to the model runner for this request."""
    _active[str(request_id)] = (np.asarray(positions, dtype=np.int64), int(matched))


def active() -> dict[str, tuple[np.ndarray, int]]:
    """Runner side: every request currently carrying a gap plan."""
    return _active


def release(request_id: str) -> None:
    """Drop a request's plan. Idempotent."""
    rid = str(request_id)
    _offered.pop(rid, None)
    _active.pop(rid, None)


def clear() -> None:
    """Forget everything. For tests, and for an engine restart in-process."""
    _offered.clear()
    _active.clear()


def stats() -> dict[str, int]:
    return {"offered": len(_offered), "active": len(_active)}
