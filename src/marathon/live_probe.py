"""Experimental: live TTFT / prompt-cache probe against the Anthropic API.

Measures, across a growing conversation, time-to-first-token and cache
read/creation token counts, to establish the Phase 0 real-world baseline:
how much does byte-stable, append-only history serialization buy from
provider prefix caching alone?

Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY`` in the
environment. Costs real (small) money; not run in CI and not covered by
tests. Usage:

    python -m marathon.live_probe --turns 6
"""

from __future__ import annotations

import argparse
import time

from .session import Session

_FILLER = (
    "This is deterministic filler content used to grow the context in a "
    "byte-stable, append-only fashion so that provider prefix caching can "
    "be measured across turns. "
) * 12

_SYSTEM = "You are a latency probe. Reply to every message with the single word: ok"


def probe(
    turns: int = 6, model: str = "claude-haiku-4-5", edit_at: int | None = None
) -> list[dict]:
    """``edit_at=N``: at turn N, mutate the first user message in place — the
    mid-session edit that invalidates naive prefix caching (Phase 1 motivation)."""
    import anthropic  # deferred: optional dependency

    client = anthropic.Anthropic()
    session = Session()
    rows: list[dict] = []

    for t in range(turns):
        if t == edit_at:
            session.edit(0, "[EDITED] " + session.messages[0]["content"])
        # Everything the model sees is decoded from the server-verified state bytes.
        state = session.turn("user", f"Turn {t}. {_FILLER} Reply 'ok'.")
        history = Session.decode(state)

        # cache_control breakpoint only on the final block (max-4-breakpoint limit).
        messages = []
        for i, h in enumerate(history):
            block: dict = {"type": "text", "text": h["content"]}
            if i == len(history) - 1:
                block["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": h["role"], "content": [block]})

        start = time.perf_counter()
        ttft = None
        with client.messages.stream(
            model=model,
            max_tokens=8,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        ) as stream:
            for _ in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - start
            final = stream.get_final_message()

        usage = final.usage
        rows.append(
            {
                "turn": t,
                "ttft_s": round(ttft, 4) if ttft is not None else None,
                "input_tokens": usage.input_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                "wire_bytes": len(session.last_payload.wire_bytes()),
                "state_bytes": len(state),
            }
        )
        session.turn("assistant", final.content[0].text)

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marathon.live_probe", description=__doc__)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5")
    parser.add_argument("--edit-at", type=int, default=None, help="mutate turn 0 at this turn")
    args = parser.parse_args(argv)
    for row in probe(turns=args.turns, model=args.model, edit_at=args.edit_at):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
