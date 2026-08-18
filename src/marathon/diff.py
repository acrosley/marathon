"""Byte-matched block delta engine (rsync-style).

Computes a delta from ``base`` to ``target`` as a sequence of ops:
- ``Copy(offset, length)``  — bytes taken verbatim from the baseline
- ``Insert(data)``          — fresh bytes present only in the target

Matching uses a rolling weak checksum (Adler-32 family) over fixed-size
blocks of the baseline, confirmed by a strong SHA-256 hash, then greedily
extended byte-by-byte. Matching is deliberately exact ("dumb and cheap");
semantic similarity belongs to a different tier of the architecture.

Wire format v0 (see docs/protocol.md):
    {"v": 0, "block_size": N, "ops": [["c", offset, length], ["i", "<base64>"]]}
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes

DEFAULT_BLOCK_SIZE = 64
_M = 1 << 16


class DeltaError(Exception):
    """Raised when a delta is malformed or cannot be applied to a baseline."""


@dataclass(frozen=True)
class Copy:
    offset: int
    length: int


@dataclass(frozen=True)
class Insert:
    data: bytes


Op = Copy | Insert


@dataclass(frozen=True)
class Delta:
    ops: tuple[Op, ...]
    block_size: int = DEFAULT_BLOCK_SIZE

    def to_dict(self) -> dict[str, Any]:
        encoded: list[list[Any]] = []
        for op in self.ops:
            if isinstance(op, Copy):
                encoded.append(["c", op.offset, op.length])
            else:
                encoded.append(["i", base64.b64encode(op.data).decode("ascii")])
        return {"v": 0, "block_size": self.block_size, "ops": encoded}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Delta:
        if d.get("v") != 0:
            raise DeltaError(f"unsupported delta version: {d.get('v')!r}")
        ops: list[Op] = []
        for raw in d["ops"]:
            kind = raw[0]
            if kind == "c":
                ops.append(Copy(int(raw[1]), int(raw[2])))
            elif kind == "i":
                ops.append(Insert(base64.b64decode(raw[1])))
            else:
                raise DeltaError(f"unknown op kind: {kind!r}")
        return cls(tuple(ops), int(d["block_size"]))

    def wire_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def insert_bytes(self) -> int:
        return sum(len(op.data) for op in self.ops if isinstance(op, Insert))

    @property
    def copy_bytes(self) -> int:
        return sum(op.length for op in self.ops if isinstance(op, Copy))


def _weak(data: bytes) -> tuple[int, int]:
    a = 0
    b = 0
    for x in data:
        a = (a + x) % _M
        b = (b + a) % _M
    return a, b


def compute_delta(base: bytes, target: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> Delta:
    """Compute a byte-exact delta such that ``apply_delta(base, delta) == target``."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    ops: list[Op] = []
    buf = bytearray()

    def flush() -> None:
        if buf:
            ops.append(Insert(bytes(buf)))
            buf.clear()

    n = len(target)
    if not base or n < block_size:
        if target:
            ops.append(Insert(target))
        return Delta(tuple(ops), block_size)

    # Index baseline blocks: weak checksum -> [(offset, strong hash)].
    table: dict[int, list[tuple[int, bytes]]] = {}
    for off in range(0, len(base) - block_size + 1, block_size):
        block = base[off : off + block_size]
        a, b = _weak(block)
        table.setdefault((b << 16) | a, []).append(
            (off, hashlib.sha256(block).digest())
        )

    i = 0
    a, b = _weak(target[0:block_size])
    while i + block_size <= n:
        match_off = -1
        candidates = table.get((b << 16) | a)
        if candidates:
            strong = hashlib.sha256(target[i : i + block_size]).digest()
            for off, s in candidates:
                if s == strong:
                    match_off = off
                    break
        if match_off >= 0:
            flush()
            length = block_size
            while (
                match_off + length < len(base)
                and i + length < n
                and base[match_off + length] == target[i + length]
            ):
                length += 1
            ops.append(Copy(match_off, length))
            i += length
            if i + block_size <= n:
                a, b = _weak(target[i : i + block_size])
        else:
            old = target[i]
            buf.append(old)
            if i + block_size < n:
                new = target[i + block_size]
                a = (a - old + new) % _M
                b = (b - block_size * old + a) % _M
            i += 1

    buf.extend(target[i:])
    flush()
    return Delta(tuple(ops), block_size)


def apply_delta(base: bytes, delta: Delta) -> bytes:
    """Reconstruct the target from a baseline and a delta."""
    out = bytearray()
    for op in delta.ops:
        if isinstance(op, Copy):
            if op.offset < 0 or op.length < 0 or op.offset + op.length > len(base):
                raise DeltaError(
                    f"copy out of range: offset={op.offset} length={op.length} "
                    f"base={len(base)}"
                )
            out += base[op.offset : op.offset + op.length]
        else:
            out += op.data
    return bytes(out)
