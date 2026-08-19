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

# One code planted before each edit turn, so the last turn's answer proves that every
# edit's re-rotated reuse kept the facts it stitched over -- including the codes that
# have now survived two and three separate edit turns.
CODES = ("7391-KAPPA", "5820-OMEGA", "1146-SIGMA")
QUESTION = "List the access codes you were given, in order, separated by commas."
EDIT = "[EDITED {n}] Amended note: this opening message was revised later in the session. "

COLS = ("turn", "wire_bytes", "state_bytes", "prompt_tokens", "prefill_s", "reused", "ph", "pol")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument(
        "--edit-at",
        default="9",
        help="comma-separated turns that rewrite the opening message (e.g. 8,14,20)",
    )
    p.add_argument("--session", default="demo")
    args = p.parse_args(argv)

    edit_at = [int(x) for x in str(args.edit_at).split(",") if x != ""]
    # a code goes in a few turns before each edit, so each one has to survive every
    # edit that follows it
    plant_at = {max(e - 3, 0): CODES[i % len(CODES)] for i, e in enumerate(edit_at)}

    c = Client(http(args.url))
    print(" ".join(f"{h:>14}" for h in COLS), "  reply")
    rows = []
    answer = ""
    for t in range(args.turns):
        if t in edit_at:
            c.edit(
                args.session,
                0,
                EDIT.format(n=edit_at.index(t)) + c.session(args.session).messages[0]["content"],
            )
        fact = f"The access code is {plant_at[t]}. " if t in plant_at else ""
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

    print()
    for t in edit_at:
        e = rows[t]
        print(
            f"edit turn {t:>2}: {e['prefill_s']:>7}s  {e['reused_tokens']:>6} tokens reused  "
            f"({e['policy']}: {e['reason']})"
        )
    planted = [plant_at[t] for t in sorted(plant_at)]
    missing = [code for code in planted if code not in answer]
    print(f"answer: {answer.strip()!r}")
    print(f"expecting: {', '.join(planted)}")
    if missing:
        print(f"FAIL: {len(missing)} planted fact(s) lost across the edit turns: {missing}")
        return 1
    print(f"PASS: all {len(planted)} planted facts answered after {len(edit_at)} edit turn(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
