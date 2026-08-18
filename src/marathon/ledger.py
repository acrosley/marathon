"""Append-only, hash-chained ledger of state snapshots.

Each snapshot commits to its state's content hash and to the previous
snapshot's chain hash, so any replica can verify the full history and any
tampering is detectable. The ledger is the deterministic "source of unchanged
immediate truth" that turn deltas are computed against.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, digest


class LedgerError(Exception):
    """Raised when a ledger fails verification or loading."""


@dataclass(frozen=True)
class Snapshot:
    index: int
    parent: str | None
    state_hash: str
    chain_hash: str
    state: Any

    @staticmethod
    def compute_chain_hash(index: int, parent: str | None, state_hash: str) -> str:
        return digest(canonical_bytes({"index": index, "parent": parent, "state_hash": state_hash}))


class Ledger:
    """In-memory append-only ledger with JSONL persistence."""

    def __init__(self) -> None:
        self._snapshots: list[Snapshot] = []

    def append(self, state: Any) -> Snapshot:
        parent = self._snapshots[-1].chain_hash if self._snapshots else None
        index = len(self._snapshots)
        state_hash = digest(canonical_bytes(state))
        chain_hash = Snapshot.compute_chain_hash(index, parent, state_hash)
        snap = Snapshot(index, parent, state_hash, chain_hash, state)
        self._snapshots.append(snap)
        return snap

    @property
    def head(self) -> Snapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, index: int) -> Snapshot:
        return self._snapshots[index]

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self._snapshots)

    def verify(self) -> None:
        """Recompute every hash in the chain; raise ``LedgerError`` on mismatch."""
        parent: str | None = None
        for i, snap in enumerate(self._snapshots):
            state_hash = digest(canonical_bytes(snap.state))
            if snap.index != i:
                raise LedgerError(f"snapshot {i}: bad index {snap.index}")
            if snap.parent != parent:
                raise LedgerError(f"snapshot {i}: broken parent link")
            if snap.state_hash != state_hash:
                raise LedgerError(f"snapshot {i}: state hash mismatch")
            expected = Snapshot.compute_chain_hash(i, parent, state_hash)
            if snap.chain_hash != expected:
                raise LedgerError(f"snapshot {i}: chain hash mismatch")
            parent = snap.chain_hash

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for snap in self._snapshots:
                record = {
                    "index": snap.index,
                    "parent": snap.parent,
                    "state_hash": snap.state_hash,
                    "chain_hash": snap.chain_hash,
                    "state": snap.state,
                }
                f.write(canonical_bytes(record).decode("utf-8") + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Ledger:
        ledger = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ledger._snapshots.append(
                    Snapshot(r["index"], r["parent"], r["state_hash"], r["chain_hash"], r["state"])
                )
        ledger.verify()
        return ledger
