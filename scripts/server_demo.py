"""End-to-end demo client: 12 turns with a mid-history edit, over the HTTP endpoint.

Run against a live ``python -m marathon.server``. The conversation plants a unique
access code in turn 3, grows append-only to turn 9 where it *rewrites turn 0's message*,
and asks for the code on the last turn. The point of the last turn is the assertion:
after an edit the model still answers from KV that was copied and re-rotated, not
recomputed, so a wrong or empty answer means the reuse lost information.

Usage: python scripts/server_demo.py --url http://127.0.0.1:8000 [--turns 12]
"""

from __future__ import annotations

import argparse
import sys

from marathon.client import Client, http

FILLER = (
    "This is deterministic filler content used to grow the context in a "
    "byte-stable, append-only fashion so that KV reuse can be measured across turns. "
) * 20

CODE = "7391-KAPPA"
PLANT_AT = 3
QUESTION = "What is the access code? Answer with only the code."
EDIT = "[EDITED] Amended note: this opening message was revised later in the session. "

COLS = ("turn", "wire_bytes", "state_bytes", "prompt_tokens", "prefill_s", "reused", "ph", "pol")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--edit-at", type=int, default=9)
    p.add_argument("--session", default="demo")
    args = p.parse_args(argv)

    c = Client(http(args.url))
    print(" ".join(f"{h:>14}" for h in COLS), "  reply")
    rows = []
    answer = ""
    for t in range(args.turns):
        if t == args.edit_at:
            # rewrite turn 0's message: every later turn's KV must be shifted, not redone
            c.edit(args.session, 0, EDIT + c.session(args.session).messages[0]["content"])
        fact = f"The access code is {CODE}. " if t == PLANT_AT else ""
        ask = t == args.turns - 1
        r = c.turn(
            args.session,
            f"Turn {t}. {fact}{FILLER} {QUESTION if ask else 'Reply ok.'}",
        )
        rows.append(r)
        print(
            " ".join(
                f"{v:>14}"
                for v in (
                    t,
                    r["wire_bytes"],
                    r["state_bytes"],
                    r["prompt_tokens"],
                    r["prefill_s"],
                    r["reused_tokens"],
                    r["phases"],
                    r["policy"],
                )
            ),
            f"  {r['reply'].strip()[:40]!r}",
            flush=True,
        )
        if ask:
            answer = r["reply"]

    edit = rows[args.edit_at]
    print(
        f"\nedit turn {args.edit_at}: {edit['prefill_s']}s, {edit['reused_tokens']} tokens reused"
    )
    print(f"  plan: {edit['policy']} / {edit['reason']}")
    print(f"answer: {answer.strip()!r} (expecting {CODE})")
    if CODE not in answer:
        print("FAIL: the planted fact did not survive the edit turn's KV reuse")
        return 1
    print("PASS: planted fact answered correctly after a mid-history edit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
