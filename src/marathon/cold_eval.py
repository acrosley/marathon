"""Phase 2 eval: does the cold tier bound the window without losing the answers?

Three exit criteria, three measurements, one script.

1. **Bounded active window on unbounded sessions.** Sessions of 60-80 turns (~30-40k
   tokens) are driven turn by turn through :class:`marathon.server.MarathonServer`;
   ``active_tokens`` is recorded per turn. Paged conditions must go flat while the
   reference grows linearly.
2. **Recall-on-miss restores demoted content.** Facts are planted across the history,
   then asked about *after* the early ones have been paged out. Exact match on the
   planted code is the score; promotion precision/recall says whether the *right*
   message came back.
3. **Quality delta vs full-context replay within tolerance.** The ``full`` condition is
   the reference: same history, no paging, every token in the window.

Conditions (all share one vLLM engine; only the server wrapper differs). The first
three run on plain vLLM prefix caching so that they differ *only* in the paging:

    full            reference: no paging, whole history in the window
    cold-norecall   paging on, recall off -- stubs only. The naive baseline: this is
                    what a window cap alone costs you.
    cold-recall     paging on, recall-on-miss on (exact + query triggers). The claim.
    cold-shift      the same policy *with* the shift connector on. A promotion is a
                    mid-history grow edit, which is exactly what position-shifted reuse
                    is for, so this is where the two phases should compose. Not in the
                    default condition set -- pass it explicitly.

    python -m marathon.cold_eval --model Qwen/Qwen3-14B-FP8 --sessions 20

Needs a GPU and weights; ``tests/test_cold.py`` covers the policy itself on CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .kvshift_eval import (
    _GREEK,
    _NOUNS,
    _SENTENCES,
    _USER_ASK,
    SNAPSHOT,
    _body,
    load_corpus,
)
from .session import Session

#: planted facts per session, spread evenly from the start of the history to the end
FACTS_PER_SESSION = 6
#: turns kept out of the fact positions at each end
FACT_MARGIN = 2


@dataclass
class EvalSession:
    """One built history plus the questions asked of it."""

    sid: int
    messages: list[dict]  # alternating user/assistant after a system prompt
    facts: list[dict]  # {noun, code, turn, msg_index}
    questions: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def build_session(
    sid: int,
    corpus: dict,
    seed: int,
    turns: int,
    count_tokens=None,
) -> EvalSession:
    """A 60-80 turn session with facts planted from the earliest turns to the latest.

    The early facts are the point: by the time the questions are asked they are far
    outside an 8k active window, so answering them requires recall, not luck.
    """
    count = count_tokens or (lambda s: len(s) // 3)
    rng = random.Random((seed * 1_000_003) ^ (sid * 7919))
    family = ["code", "prose", "qa"][sid % 3]
    system = (
        "You are a meticulous project assistant reading a long working log. "
        "When asked for a code, reply with the code and nothing else."
    )
    messages: list[dict] = [{"role": "system", "content": system, "governing": True}]

    nouns = rng.sample(_NOUNS, FACTS_PER_SESSION)
    span = range(FACT_MARGIN, turns - FACT_MARGIN)
    step = max(1, len(span) // FACTS_PER_SESSION)
    fact_turns = [span[min(i * step, len(span) - 1)] for i in range(FACTS_PER_SESSION)]
    facts = [
        {"noun": n, "code": f"{rng.randint(1000, 9999)}-{rng.choice(_GREEK)}", "turn": t}
        for n, t in zip(nouns, fact_turns, strict=True)
    ]
    by_turn = {f["turn"]: f for f in facts}

    total = count(system)
    for t in range(turns):
        name, body = _body(rng, family, corpus)
        planted = ""
        if t in by_turn:
            f = by_turn[t]
            planted = f"\nThe {f['noun']} code is {f['code']}."
            f["msg_index"] = len(messages)
        filler = _SENTENCES[(t + sid) % len(_SENTENCES)]
        ask = _USER_ASK[family][t % 3]
        head = f"Entry {t}. {filler} {ask}"
        # The planted fact goes *after* the body, never in the opening words: a stub
        # keeps the head of the message, so a fact planted in the head would be readable
        # straight off the stub and the eval would score a recall it never performed.
        messages.append({"role": "user", "content": f"{head}\n{body}{planted}"})
        messages.append(
            {
                "role": "assistant",
                "content": f"Noted entry {t} ({name}). {rng.choice(_SENTENCES)}",
            }
        )
        total += count(body) + 60

    item = EvalSession(sid, messages, facts)
    item.meta = {"family": family, "turns": turns, "approx_tokens": total}
    item.questions = _questions(rng, facts, turns)
    return item


def _questions(rng: random.Random, facts: list[dict], turns: int) -> list[dict]:
    """Targeted questions about old content, about recent content, and distractors.

    ``kind`` is what the row is scored as: ``old`` facts are the ones paging removes,
    ``recent`` facts should be answerable from the active window alone (a control that
    catches a policy that has simply broken the model), and ``distractor`` asks for a
    code that was never planted -- a fabricated answer there is a false positive, and
    any promotion it triggers is a precision miss.
    """
    old = [f for f in facts if f["turn"] < turns * 0.5]
    recent = [f for f in facts if f["turn"] >= turns * 0.5]
    out = [
        {
            "kind": "old" if f in old else "recent",
            "noun": f["noun"],
            "expected": f["code"],
            "target": f["msg_index"],
            "text": f"What is the {f['noun']} code? Reply with the code and nothing else.",
        }
        for f in old + recent
    ]
    unseen = [n for n in _NOUNS if n not in {f["noun"] for f in facts}]
    for noun in rng.sample(unseen, 2):
        out.append(
            {
                "kind": "distractor",
                "noun": noun,
                "expected": None,
                "target": None,
                "text": (
                    f"What is the {noun} code? If it was never mentioned, "
                    'reply exactly "not mentioned".'
                ),
            }
        )
    return out


# ----------------------------------------------------------------------- scoring


def extract_code(text: str) -> str | None:
    """The first ``NNNN-GREEK`` token in a reply, or None."""
    import re

    m = re.search(r"\b(\d{4})-(" + "|".join(_GREEK) + r")\b", text.upper())
    return f"{m.group(1)}-{m.group(2)}" if m else None


def score(question: dict, reply: str) -> dict:
    """Exact match for a fact question; fabrication check for a distractor."""
    got = extract_code(reply)
    if question["kind"] == "distractor":
        return {"correct": got is None, "fabricated": got is not None, "answer": got}
    return {"correct": got == question["expected"], "fabricated": False, "answer": got}


# -------------------------------------------------------------------- the driver


def drive(
    server,
    session_id: str,
    item: EvalSession,
    max_history_tokens: int = 1,
    generate_history: bool = False,
) -> list[dict]:
    """Replay the history turn by turn, then ask the questions. One row per turn.

    Assistant replies are scripted rather than generated, so every condition sees a
    byte-identical history and differences in the answers can only come from the paging
    policy.

    ``generate_history`` decides whether the history turns hit the engine. They do not
    need to: what a history turn contributes is a *paging decision* and a window size,
    both of which are computed before any prefill, and the questions are asked against
    exactly the same state either way. Prefilling them anyway costs a full recompute per
    turn -- paging edits the front of the view every turn, so prefix caching cannot help
    -- which is ~600 s per session against ~5 s for the questions alone. Turn it on to
    measure per-turn serving cost; leave it off to measure the policy and the answers.
    """
    sess = Session()
    rows: list[dict] = []
    pairs = [(i, m) for i, m in enumerate(item.messages)]
    server.max_tokens = max_history_tokens
    for i, message in pairs:
        if message["role"] == "assistant":
            sess.messages.append(dict(message))  # scripted, no round trip (as a client does)
            continue
        sess.turn(message["role"], message["content"], governing=message.get("governing"))
        out = server.turn(session_id, sess.last_payload.to_dict(), generate=generate_history)
        rows.append(
            {
                "phase": "history",
                "msg_index": i,
                "generated": generate_history,
                **_metrics(out),
            }
        )

    server.max_tokens = 24
    for q in item.questions:
        sess.turn("user", q["text"])
        out = server.turn(session_id, sess.last_payload.to_dict())
        sess.messages.append({"role": "assistant", "content": out["reply"]})
        promoted = [p["index"] for p in out["promotions"]]
        rows.append(
            {
                "phase": "question",
                "kind": q["kind"],
                "noun": q["noun"],
                "expected": q["expected"],
                "target": q["target"],
                "target_was_cold": q["target"] is not None and q["target"] in out["cold_before"],
                "promoted": promoted,
                "promoted_target": q["target"] in promoted if q["target"] is not None else None,
                "reply": out["reply"].strip()[:120],
                **score(q, out["reply"]),
                **_metrics(out),
            }
        )
    return rows


def _metrics(out: dict) -> dict:
    return {
        "active_tokens": out["active_tokens"],
        "prefill_s": out["prefill_s"],
        "cold_count": out["cold_count"],
        "reused_tokens": out["reused_tokens"],
        "policy": out["policy"],
        "n_promotions": len(out["promotions"]),
        "n_demotions": len(out["demotions"]),
    }


# ---------------------------------------------------------------------- reporting


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def _median(xs) -> float:
    xs = list(xs)
    return statistics.median(xs) if xs else float("nan")


def summarise(rows: list[dict]) -> list[dict]:
    """Per-condition table: window, cost, accuracy, promotion quality."""
    out = []
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        hist = [r for r in rs if r["phase"] == "history"]
        qs = [r for r in rs if r["phase"] == "question"]
        tail = hist[len(hist) // 2 :]  # second half: where paging is in steady state
        facts = [r for r in qs if r["kind"] in ("old", "recent")]
        old = [r for r in facts if r["kind"] == "old"]
        recent = [r for r in facts if r["kind"] == "recent"]
        dis = [r for r in qs if r["kind"] == "distractor"]
        # a promotion is "right" if it brought back the message holding the answer
        needed = [r for r in facts if r["target_was_cold"]]
        promotions = sum(r["n_promotions"] for r in qs)
        hits = sum(1 for r in needed if r["promoted_target"])
        out.append(
            {
                "condition": cond,
                "n_sessions": len({r["sid"] for r in rs}),
                "turns": len(hist),
                "active_p50": _median(r["active_tokens"] for r in tail),
                "active_max": max((r["active_tokens"] for r in hist), default=0),
                "prefill_p50": round(_median(r["prefill_s"] for r in tail), 4),
                "prefill_max": round(max((r["prefill_s"] for r in hist), default=0), 4),
                "em_old": round(_mean(r["correct"] for r in old), 4),
                "em_recent": round(_mean(r["correct"] for r in recent), 4),
                "em_all": round(_mean(r["correct"] for r in facts), 4),
                "fabricated": round(_mean(r["fabricated"] for r in dis), 4),
                "n_cold_q": len(needed),
                "promo_recall": round(hits / len(needed), 4) if needed else float("nan"),
                "promo_precision": round(hits / promotions, 4) if promotions else float("nan"),
                "q_prefill_p50": round(_median(r["prefill_s"] for r in qs), 4),
                "promo_prefill_p50": round(
                    _median(r["prefill_s"] for r in qs if r["n_promotions"]), 4
                ),
            }
        )
    return out


def print_table(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    widths = [max(len(k), *(len(str(r[k])) for r in rows)) for k in keys]
    print(f"\n{title}")
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w) for k, w in zip(keys, widths, strict=True)))


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marathon.cold_eval", description="Phase 2 cold-tier eval")
    p.add_argument("--model", default="Qwen/Qwen3-14B-FP8")
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--turns", type=int, default=70)
    p.add_argument("--active-window", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=40960)
    # 0.93, not the usual 0.85: the shift store is carved out of the same budget *before*
    # vLLM sizes its own KV cache, and CUDA-graph profiling takes another ~2.8 GB, so
    # 0.85 with a 24k store leaves vLLM negative KV cache and the engine refuses to start.
    p.add_argument("--gpu-util", type=float, default=0.93)
    # 24k tokens of shift store is ~4 GB on Qwen3-14B (164 KB/token). 64k would be
    # 10.7 GB out of a 27 GB budget, which starves vLLM's own KV cache and puts the
    # engine into preemption -- and preemption interleaved with a connector load is
    # exactly the interaction protocol.md lists as untested.
    p.add_argument("--store-tokens", type=int, default=24576)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--out", default="cold_eval.jsonl")
    # the frozen sample, not the working tree: the corpus is built from repo source, so
    # editing any file in the repo (these docs included) would silently move every
    # session and make two runs of different conditions incomparable
    p.add_argument("--corpus", choices=("snapshot", "worktree"), default="snapshot")
    p.add_argument(
        "--generate-history",
        action="store_true",
        help="prefill every history turn too (measures per-turn serving cost; ~100x "
        "slower, since paging defeats prefix caching on every turn)",
    )
    p.add_argument(
        "--conditions",
        default="full,cold-norecall,cold-recall",
        help="comma-separated subset to run",
    )
    p.add_argument(
        "--from-jsonl",
        nargs="+",
        help="re-print the table from previous runs' JSONL instead of running "
        "(sessions are seeded, so runs of different conditions merge)",
    )
    args = p.parse_args(argv)

    if args.from_jsonl:
        rows = [
            json.loads(line)
            for path in args.from_jsonl
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print_table("Phase 2 cold tier", summarise(rows))
        print(f"{len(rows)} rows from {len(args.from_jsonl)} file(s)")
        return 0

    from .cold import TransformerEmbedder
    from .server import ChatTokenizer, MarathonServer, VllmEngine

    tok = ChatTokenizer(args.model)
    count = lambda s: len(tok.encode(s))  # noqa: E731
    corpus = load_corpus(SNAPSHOT if args.corpus == "snapshot" else None)
    items = [
        build_session(sid, corpus, args.seed, args.turns, count) for sid in range(args.sessions)
    ]
    sizes = [sum(count(m["content"]) for m in it.messages) for it in items]
    print(
        f"{len(items)} sessions, {args.turns} turns, "
        f"history tokens p50={_median(sizes):.0f} min={min(sizes)} max={max(sizes)}",
        flush=True,
    )

    engine = VllmEngine(args.model, args.max_model_len, args.gpu_util, args.store_tokens)
    embedder = TransformerEmbedder(args.embed_model)
    cold_kwargs = {"embedder": embedder, "threshold": args.threshold, "top_k": args.top_k}
    specs = {
        # the reference is full-context serving as an ordinary system does it: plain
        # vLLM prefix caching, no connector. Leaving the shift store attached makes it
        # save a 35k-token prompt into a 24k store every turn, which is pure eviction
        # churn -- p50 prefill 2.15 s instead of 0.07 s, at 123 W of 575 W (memory bound,
        # measured 2026-08-19) -- and measures the store, not the baseline.
        "full": {"reuse": False},
        # The paged conditions run connector-off too, for the same reason: the cold tier
        # is a client-side policy and the three exit criteria are about the policy, so
        # the reference and the paged conditions must differ *only* in the paging.
        "cold-norecall": {
            "active_window": args.active_window,
            "cold_kwargs": {"recall": False},
            "reuse": False,
        },
        "cold-recall": {
            "active_window": args.active_window,
            "cold_kwargs": cold_kwargs,
            "reuse": False,
        },
        # The interaction probe: the same policy *with* position-shifted KV reuse. A
        # promotion is a mid-history grow edit, which is what shifted reuse exists for,
        # so this is where Phase 1 and Phase 2 should compose. Not in the default set.
        "cold-shift": {
            "active_window": args.active_window,
            "cold_kwargs": cold_kwargs,
            "reuse": True,
        },
    }

    rows: list[dict] = []
    out_path = Path(args.out)
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for cond in args.conditions.split(","):
            server = MarathonServer(engine=engine, tokenizer=tok, **specs[cond])
            for item in items:
                t0 = time.time()
                for row in drive(
                    server, f"{cond}-{item.sid}", item, generate_history=args.generate_history
                ):
                    row.update(condition=cond, sid=item.sid, family=item.meta["family"])
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[{cond}] session {item.sid} done in {time.time() - t0:.1f}s "
                    f"({time.time() - started:.0f}s total)",
                    flush=True,
                )

    print_table("Phase 2 cold tier", summarise(rows))
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
