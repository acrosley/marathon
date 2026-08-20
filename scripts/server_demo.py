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


def code_for(t: int) -> str:
    """A distinct, tokenizer-friendly code per turn."""
    return f"{7000 + 13 * t}-{['KAPPA', 'OMEGA', 'SIGMA', 'DELTA', 'THETA'][t % 5]}"


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
    p.add_argument(
        "--demote",
        type=int,
        default=0,
        help="cold-tier shape: from this turn on, stub the oldest live message every "
        "turn (a front-of-view shrink edit on every single turn)",
    )
    p.add_argument("--json", default=None, help="write per-turn rows here for comparison")
    p.add_argument(
        "--fact-probe",
        action="store_true",
        help="demote mode: plant a code every turn and ask for the one from two turns "
        "back, so each turn scores exact-match on a fact that lives in the reused span",
    )
    p.add_argument(
        "--fixed-replies",
        action="store_true",
        help="append a canned assistant message instead of the generated one, so two "
        "runs keep byte-identical histories and each turn is an independent comparison",
    )
    args = p.parse_args(argv)

    edit_at = [int(x) for x in str(args.edit_at).split(",") if x != ""]
    if args.demote or args.fact_probe:
        edit_at = []  # paging drives the edits instead
    # a code goes in a few turns before each edit, so each one has to survive every
    # edit that follows it
    plant_at = {max(e - 3, 0): CODES[i % len(CODES)] for i, e in enumerate(edit_at)}

    c = Client(http(args.url))
    print(" ".join(f"{h:>14}" for h in COLS), "  reply")
    rows = []
    answer = ""
    for t in range(args.turns):
        if args.demote and t >= args.demote:
            # the cold tier's demotion: the oldest live message becomes a stub carrying
            # its own content address. A shrink edit at the front of the view, every turn.
            d = t - args.demote
            c.edit(args.session, 2 * d, f"[cold #{d} {d:08x}]")
        if t in edit_at:
            c.edit(
                args.session,
                0,
                EDIT.format(n=edit_at.index(t)) + c.session(args.session).messages[0]["content"],
            )
        fact = f"The access code is {plant_at[t]}. " if t in plant_at else ""
        ask = t == args.turns - 1
        # In demote mode every turn asks something that can only be answered by
        # attending over the *reused* span. "Reply ok." would compare 'Ok.' against
        # 'Ok.' and call a corrupted cache a match.
        want = None
        if args.fact_probe:
            # One code alive at a time, planted every 4th turn. The question is asked on
            # the *two* turns after it, not one: with a reuse/refresh alternation every
            # other turn, a single fixed offset locks every scored turn onto the same
            # phase, and the 2026-08-19 14B run scored 7 reuse turns and 0 refresh turns
            # without noticing. Asking twice covers both parities.
            if t % 4 == 0:
                fact = f"The access code is {code_for(t)}. "
                request = "Reply ok."
            elif t % 4 in (1, 2) and t >= 1:
                want = code_for(t - (t % 4))
                request = "What is the access code? Answer with only the code."
            else:
                request = "Reply ok."
        elif ask:
            request = QUESTION
        elif args.demote:
            request = "In one sentence, summarize everything you have been told so far."
        else:
            request = "Reply ok."
        r = c.turn(args.session, f"Turn {t}. {fact}{FILLER} {request}")
        r["want"] = want
        r["hit"] = None if want is None else (want in r["reply"])
        if args.fixed_replies:
            # Teacher forcing. Without it the first differing reply makes the two runs
            # diverge as *conversations*, and every later mismatch is an echo of that
            # one rather than a fresh signal about the KV.
            c.session(args.session).messages[-1]["content"] = "Understood."
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

    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"turns": args.turns, "demote": args.demote, "rows": rows}, f, indent=1)

    print()
    if args.fact_probe:
        scored = [r for r in rows if r["hit"] is not None]
        hits = sum(1 for r in scored if r["hit"])
        print(f"fact exact-match: {hits}/{len(scored)} = {hits / max(len(scored), 1):.3f}")
    if args.demote or args.fact_probe:
        edited = [r for r in rows if r["reused_tokens"] > 0]
        pre = [r["prefill_s"] for r in rows]
        pre_sorted = sorted(pre)
        print(
            f"demote mode: {len(edited)}/{len(rows)} turns reused, "
            f"prefill p50={pre_sorted[len(pre) // 2]:.3f}s max={max(pre):.3f}s"
        )
        if edited:
            deltas = [d for r in edited for d in r["deltas"]]
            print(f"  segment deltas: min={min(deltas)} max={max(deltas)}")
    for t in edit_at:
        e = rows[t]
        print(
            f"edit turn {t:>2}: {e['prefill_s']:>7}s  {e['reused_tokens']:>6} tokens reused  "
            f"({e['policy']}: {e['reason']})"
        )
    if (args.demote or args.fact_probe) and not plant_at:
        return 0
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
