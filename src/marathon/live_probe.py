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

_FILLER = (
    "This is deterministic filler content used to grow the context in a "
    "byte-stable, append-only fashion so that provider prefix caching can "
    "be measured across turns. "
) * 12

_SYSTEM = "You are a latency probe. Reply to every message with the single word: ok"


def probe(turns: int = 6, model: str = "claude-3-5-haiku-latest") -> list[dict]:
    import anthropic  # deferred: optional dependency

    client = anthropic.Anthropic()
    history: list[dict] = []
    rows: list[dict] = []

    for t in range(turns):
        history.append({"role": "user", "text": f"Turn {t}. {_FILLER} Reply 'ok'."})

        # Rebuild message list each turn: cache_control breakpoint only on the
        # final block (max-4-breakpoint limit), earlier turns byte-stable.
        messages = []
        for i, h in enumerate(history):
            block: dict = {"type": "text", "text": h["text"]}
            if i == len(history) - 1 and h["role"] == "user":
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
            }
        )
        history.append({"role": "assistant", "text": final.content[0].text})

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marathon.live_probe", description=__doc__)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--model", type=str, default="claude-3-5-haiku-latest")
    args = parser.parse_args(argv)
    for row in probe(turns=args.turns, model=args.model):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
