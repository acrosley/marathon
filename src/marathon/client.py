"""The client half: keep the history, ship deltas, collect replies.

The client is the party that *owns* the conversation. It holds the full logical
history — its own messages and the server's replies — and each turn it sends only the
delta from the last state the server acknowledged, plus the new input. That is the
whole point of the protocol, so the client is deliberately tiny:
:class:`marathon.session.Session` already does the delta bookkeeping, and this module
adds only the transport and the reply plumbing.

Assistant replies are appended to the local history *without* a protocol round trip.
They are not a state the server acknowledged, so they must not advance the baseline;
they simply become part of the next turn's delta, which is why a normal append-only
turn ships two messages' worth of bytes and nothing else.

Two transports, same one-function interface ``send(session_id, payload_dict) -> dict``:

    client = Client(local(server))                     # in-process, no sockets
    client = Client(http("http://127.0.0.1:8000"))     # over the HTTP endpoint

An edit is :meth:`Client.edit`, which mutates an earlier message in place; it takes
effect on the next turn, where the delta engine will see it and the server's reuse plan
will decide what KV survives it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .diff import DEFAULT_BLOCK_SIZE
from .session import Session

Transport = Callable[[str, dict], dict]


def local(server: Any) -> Transport:
    """In-process transport: call :meth:`marathon.server.MarathonServer.turn` directly."""

    def send(session_id: str, payload: dict) -> dict:
        return server.turn(session_id, payload)

    return send


def http(url: str, timeout: float = 600.0) -> Transport:
    """HTTP transport against ``POST <url>/v1/turn``."""
    endpoint = url.rstrip("/") + "/v1/turn"

    def send(session_id: str, payload: dict) -> dict:
        body = json.dumps({"session": session_id, "payload": payload}).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # the server answers rejections as JSON; a bare "HTTP 400" hides the reason
            return json.loads(e.read().decode("utf-8"))

    return send


class Client:
    """A conversation client over one transport, with as many sessions as you like."""

    def __init__(self, send: Transport, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.send = send
        self.block_size = block_size
        self.sessions: dict[str, Session] = {}

    def session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session(self.block_size))

    def edit(self, session_id: str, index: int, content: str) -> None:
        """Rewrite an earlier message; it ships with the next turn's delta."""
        self.session(session_id).edit(index, content)

    def turn(self, session_id: str, text: str, role: str = "user", **kw: Any) -> dict:
        """Send one turn and return the server's response dict (``reply`` plus metrics).

        The baseline advances when the payload is built, so a server that *rejects* a
        payload leaves this client's baseline ahead of the server's: v1 expects the
        caller to drop the session and start over, which is also the documented recovery
        for an unknown baseline.
        """
        session = self.session(session_id)
        session.turn(role, text, **kw)
        assert session.last_payload is not None
        response = self.send(session_id, session.last_payload.to_dict())
        if "reply" not in response:
            raise RuntimeError(f"server rejected the turn: {response}")
        session.messages.append({"role": "assistant", "content": response["reply"]})
        return response
