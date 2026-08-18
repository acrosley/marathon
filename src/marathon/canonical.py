"""Canonical, byte-stable serialization primitives.

Determinism is the load-bearing property of Marathon (see DESIGN.md): two
parties that hold the same logical state must derive byte-identical
serializations, so state can be referenced by hash instead of retransmitted.

Rules enforced here:
- JSON with sorted keys, minimal separators, UTF-8, NaN/Infinity forbidden.
- History serialization is append-only: serializing ``history[:k]`` always
  yields a byte-prefix of serializing ``history[:k+1]``. This is what makes
  provider prefix caching hit maximally (Phase 0 strategy).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical, byte-stable JSON (UTF-8)."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    """Content address for a byte string: ``sha256:<hex>``."""
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def snapshot_hash(obj: Any) -> str:
    """Content address for a JSON-serializable object."""
    return digest(canonical_bytes(obj))


def serialize_history(entries: list[Any]) -> bytes:
    """Serialize a history as canonical JSON Lines.

    Append-only guarantee: for any ``k``, ``serialize_history(entries[:k])``
    is a byte-prefix of ``serialize_history(entries)``.
    """
    out = bytearray()
    for entry in entries:
        out += canonical_bytes(entry)
        out += b"\n"
    return bytes(out)
