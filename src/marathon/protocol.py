"""Turn protocol: ``{baseline_hash, target_hash, delta, new_input}``.

The client diffs its new state against the last acknowledged baseline and
sends only the delta plus fresh input. The server resolves the baseline from
its content-addressed store, applies the delta, and verifies the result
against ``target_hash`` before trusting it — reconstruction is never assumed,
always proven. A verified target becomes the next baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, digest
from .diff import DEFAULT_BLOCK_SIZE, Delta, apply_delta, compute_delta

PROTOCOL_VERSION = 0


class ProtocolError(Exception):
    """Base class for turn-protocol failures."""


class UnknownBaselineError(ProtocolError):
    """The referenced baseline hash is not present in the store."""


class IntegrityError(ProtocolError):
    """Applying the delta did not reproduce the declared target hash."""


class BaselineStore:
    """Content-addressed store of baseline byte strings."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        h = digest(data)
        self._store[h] = data
        return h

    def get(self, h: str) -> bytes:
        try:
            return self._store[h]
        except KeyError:
            raise UnknownBaselineError(h) from None

    def __contains__(self, h: str) -> bool:
        return h in self._store

    def __len__(self) -> int:
        return len(self._store)


@dataclass(frozen=True)
class TurnPayload:
    baseline_hash: str | None  # None on the first turn (empty baseline)
    target_hash: str
    delta: Delta
    new_input: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": PROTOCOL_VERSION,
            "baseline_hash": self.baseline_hash,
            "target_hash": self.target_hash,
            "delta": self.delta.to_dict(),
            "new_input": self.new_input,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TurnPayload:
        if d.get("v") != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {d.get('v')!r}")
        return cls(
            baseline_hash=d["baseline_hash"],
            target_hash=d["target_hash"],
            delta=Delta.from_dict(d["delta"]),
            new_input=d["new_input"],
        )

    def wire_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_wire(cls, data: bytes) -> TurnPayload:
        return cls.from_dict(json.loads(data.decode("utf-8")))


def prepare_turn(
    store: BaselineStore,
    baseline_hash: str | None,
    target: bytes,
    new_input: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> TurnPayload:
    """Client side: compute the delta from the acknowledged baseline to ``target``."""
    base = store.get(baseline_hash) if baseline_hash is not None else b""
    delta = compute_delta(base, target, block_size)
    return TurnPayload(baseline_hash, digest(target), delta, new_input)


def resolve_turn(store: BaselineStore, payload: TurnPayload) -> bytes:
    """Server side: reconstruct, verify, and promote the target to next baseline."""
    base = store.get(payload.baseline_hash) if payload.baseline_hash is not None else b""
    result = apply_delta(base, payload.delta)
    if digest(result) != payload.target_hash:
        raise IntegrityError(
            f"reconstructed state does not match declared target "
            f"({digest(result)} != {payload.target_hash})"
        )
    store.put(result)
    return result
