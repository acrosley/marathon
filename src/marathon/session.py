"""Session runner: drive a real conversation through the ledger and turn protocol.

The canonical serializer is the single path to the wire. Whatever a model is
shown is decoded from the *server-verified* reconstruction of the state, never
from client memory — so cache behaviour measured against a provider is a
property of the library, and correctness can be checked against a full-context
replay at every turn.
"""

from __future__ import annotations

import json
from typing import Any

from .canonical import digest, serialize_history
from .diff import DEFAULT_BLOCK_SIZE
from .ledger import Ledger
from .protocol import BaselineStore, TurnPayload, prepare_turn, resolve_turn


class Session:
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.block_size = block_size
        self.messages: list[dict[str, Any]] = []  # client-side logical history
        self.store = BaselineStore()  # server-side content-addressed baselines
        self.ledger = Ledger()
        self.baseline_hash: str | None = None
        self.last_payload: TurnPayload | None = None

    def edit(self, index: int, content: str) -> None:
        """Mutate an earlier message in place (takes effect on the next turn).

        Everything but the content is preserved, ``governing`` included: the flag is a
        property of the slot, and the reuse plan reads it off the *old* state.
        """
        self.messages[index] = {**self.messages[index], "content": content}

    def turn(self, role: str, content: str, governing: bool | None = None) -> bytes:
        """Append a message, ship the delta, return the server-verified state bytes.

        ``governing`` marks a message that steers *later* generation (system prompt,
        standing instructions, persona, output format, tool policy). Editing such a
        message makes position-shifted KV reuse unsafe — see ``reuse_plan``. It defaults
        to True for the system role. The key is serialized only when True, so canonical
        bytes of every session that never sets it are unchanged.
        """
        message: dict[str, Any] = {"role": role, "content": content}
        if governing if governing is not None else role == "system":
            message["governing"] = True
        self.messages.append(message)
        state = self.replay()
        payload = prepare_turn(self.store, self.baseline_hash, state, content, self.block_size)
        resolved = resolve_turn(self.store, TurnPayload.from_wire(payload.wire_bytes()))
        self.baseline_hash = payload.target_hash
        self.ledger.append({"turn": len(self.ledger), "history_digest": digest(state)})
        self.last_payload = payload
        return resolved

    def replay(self) -> bytes:
        """Full-context replay: serialize the whole logical history from scratch."""
        return serialize_history(self.messages)

    @staticmethod
    def decode(state: bytes) -> list[dict[str, Any]]:
        """Messages as the model would see them, decoded from canonical state bytes."""
        return [json.loads(line) for line in state.split(b"\n") if line]
