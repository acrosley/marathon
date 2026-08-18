"""Offline benchmark: full-resend vs delta-encoded context over a simulated session.

Simulates a growing conversation. Each turn appends a message; optionally an
earlier message is edited every ``edit_every`` turns (the case that destroys
naive prefix caching but that block-matched deltas absorb). For every turn we
measure what the wire would carry under full resend versus the Marathon turn
payload (delta + hashes + new input).

Usage:
    python -m marathon.bench --turns 50 --growth 400 --edit-every 10
    python -m marathon.bench --json report.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import asdict, dataclass

from .canonical import digest, serialize_history
from .diff import DEFAULT_BLOCK_SIZE
from .ledger import Ledger
from .protocol import BaselineStore, prepare_turn, resolve_turn

_ALPHABET = string.ascii_letters + string.digits + "     .,;:"
_BYTES_PER_TOKEN_ESTIMATE = 4  # rough heuristic for reporting only


@dataclass(frozen=True)
class TurnMetric:
    turn: int
    state_bytes: int
    full_bytes: int
    wire_bytes: int
    delta_insert_bytes: int
    delta_copy_bytes: int
    ratio: float  # wire / full


def simulate(
    turns: int = 50,
    growth: int = 400,
    edit_every: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 7,
) -> dict:
    """Run a deterministic simulated session; return a metrics report."""
    rng = random.Random(seed)
    store = BaselineStore()
    ledger = Ledger()
    messages: list[dict] = []
    baseline_hash: str | None = None
    metrics: list[TurnMetric] = []

    for t in range(turns):
        content = "".join(rng.choice(_ALPHABET) for _ in range(growth))
        role = "user" if t % 2 == 0 else "assistant"
        messages.append({"role": role, "turn": t, "content": content})

        if edit_every and t > 0 and t % edit_every == 0:
            j = rng.randrange(len(messages) - 1)
            messages[j] = {**messages[j], "content": messages[j]["content"] + " [edited]"}

        state = serialize_history(messages)
        payload = prepare_turn(
            store, baseline_hash, state, new_input=content, block_size=block_size
        )
        resolved = resolve_turn(store, payload)  # server side: reconstruct + verify
        assert resolved == state
        baseline_hash = payload.target_hash

        ledger.append({"turn": t, "history_digest": digest(state)})

        wire = len(payload.wire_bytes())
        full = len(state)
        metrics.append(
            TurnMetric(
                turn=t,
                state_bytes=full,
                full_bytes=full,
                wire_bytes=wire,
                delta_insert_bytes=payload.delta.insert_bytes,
                delta_copy_bytes=payload.delta.copy_bytes,
                ratio=wire / full if full else 0.0,
            )
        )

    ledger.verify()
    full_total = sum(m.full_bytes for m in metrics)
    wire_total = sum(m.wire_bytes for m in metrics)
    return {
        "params": {
            "turns": turns,
            "growth": growth,
            "edit_every": edit_every,
            "block_size": block_size,
            "seed": seed,
        },
        "turns": [asdict(m) for m in metrics],
        "totals": {
            "full_bytes": full_total,
            "wire_bytes": wire_total,
            "savings_ratio": 1 - (wire_total / full_total) if full_total else 0.0,
            "full_tokens_estimate": full_total // _BYTES_PER_TOKEN_ESTIMATE,
            "wire_tokens_estimate": wire_total // _BYTES_PER_TOKEN_ESTIMATE,
        },
    }


def _print_report(report: dict) -> None:
    p = report["params"]
    print(
        f"marathon bench — turns={p['turns']} growth={p['growth']}B "
        f"edit_every={p['edit_every']} block_size={p['block_size']}"
    )
    print(f"{'turn':>5} {'state B':>10} {'full B':>10} {'wire B':>10} {'ratio':>8}")
    rows = report["turns"]
    shown = rows if len(rows) <= 12 else rows[:6] + rows[-6:]
    last_turn = None
    for m in shown:
        if last_turn is not None and m["turn"] != last_turn + 1:
            print(f"{'...':>5}")
        print(
            f"{m['turn']:>5} {m['state_bytes']:>10} {m['full_bytes']:>10} "
            f"{m['wire_bytes']:>10} {m['ratio']:>8.4f}"
        )
        last_turn = m["turn"]
    t = report["totals"]
    print(
        f"\ntotals: full={t['full_bytes']:,} B  wire={t['wire_bytes']:,} B  "
        f"savings={t['savings_ratio']:.2%}"
    )
    print(
        f"token estimate (~{_BYTES_PER_TOKEN_ESTIMATE} B/token): "
        f"full≈{t['full_tokens_estimate']:,}  wire≈{t['wire_tokens_estimate']:,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marathon.bench", description=__doc__)
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--growth", type=int, default=400, help="bytes of new content per turn")
    parser.add_argument(
        "--edit-every", type=int, default=0, help="edit an earlier message every N turns"
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--json", type=str, default=None, help="write full JSON report to this path"
    )
    args = parser.parse_args(argv)

    report = simulate(
        turns=args.turns,
        growth=args.growth,
        edit_every=args.edit_every,
        block_size=args.block_size,
        seed=args.seed,
    )
    _print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
