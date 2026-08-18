"""Marathon: delta-encoded context architecture for LLMs.

Per-turn cost proportional to what changed, not to total context size.
See DESIGN.md (doc 0001) and docs/PLAN.md.
"""

from .canonical import canonical_bytes, digest, serialize_history, snapshot_hash
from .diff import Copy, Delta, DeltaError, Insert, apply_delta, compute_delta
from .ledger import Ledger, LedgerError, Snapshot
from .protocol import (
    BaselineStore,
    IntegrityError,
    ProtocolError,
    TurnPayload,
    UnknownBaselineError,
    prepare_turn,
    resolve_turn,
)
from .session import Session

__version__ = "0.0.1"

__all__ = [
    "BaselineStore",
    "Copy",
    "Delta",
    "DeltaError",
    "Insert",
    "IntegrityError",
    "Ledger",
    "LedgerError",
    "ProtocolError",
    "Session",
    "Snapshot",
    "TurnPayload",
    "UnknownBaselineError",
    "apply_delta",
    "canonical_bytes",
    "compute_delta",
    "digest",
    "prepare_turn",
    "resolve_turn",
    "serialize_history",
    "snapshot_hash",
]
