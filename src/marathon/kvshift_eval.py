"""Distribution-level quality eval for position-shifted KV reuse.

``kvshift_probe`` answers "does re-rotated reuse work?" on six hand-built scenarios.
This answers "how often, and where, does it *not*?" over a population: N synthetic but
realistic multi-turn sessions x 5 edit kinds x a pool of query types x 4 serving
conditions, all seeded, with per-item JSONL and a summary table.

Conditions (all against the same new token sequence):

    full-recompute   reference: one forward over everything (ground truth)
    reuse-all        P reused, E' fresh, S re-rotated by delta          <- the claim
    no-rerotate      control: same reuse, S's keys left at stale angles
    prefix-equiv     what vLLM prefix caching can do: reuse P only,
                     recompute everything from the edit onward (KL ~ 0 by
                     construction; recorded for its *cost*, not its quality)

Metrics per (session, edit, query): mean/max KL over 32 teacher-forced continuation
tokens of the reference's own greedy output, first-token KL, per-position top-1
agreement, exact match of the free-running greedy answer, and fraction of tokens
forwarded.

    python -m marathon.kvshift_eval --model Qwen/Qwen3-8B --sessions 60

Not run in CI (needs a GPU and weights); ``tests/test_kvshift_eval.py`` covers the
pure session/edit construction on CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .kvshift import (
    Policy,
    byte_span,
    compare,
    prefill,
    run_full,
    run_policy,
    token_span,
)
from .session import Session

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------- corpora

_CODE_FILES = ["src/marathon/*.py", "tests/*.py"]
_PROSE_FILES = ["docs/*.md", "DESIGN.md", "README.md"]


def _read(pattern: str) -> list[tuple[str, str]]:
    out = []
    for path in sorted(REPO.glob(pattern)):
        try:
            name = str(path.relative_to(REPO)).replace("\\", "/")
            out.append((name, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
    return out


#: Frozen sample of the live corpus. The tests build sessions from this, not from the
#: working tree, so editing repo source cannot change what they assert.
SNAPSHOT = REPO / "tests" / "data" / "kvshift_eval_corpus.json"


def load_corpus(snapshot: Path | str | None = None) -> dict[str, list]:
    """Real repo text, chopped into turn-sized chunks. Deterministic (sorted paths).

    Pass ``snapshot`` to read a frozen sample instead of the working tree.
    """
    if snapshot is not None:
        data = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        return {k: [tuple(x) for x in v] for k, v in data.items()}
    code = []
    for name, text in [x for p in _CODE_FILES for x in _read(p)]:
        lines = text.splitlines()
        for i in range(0, len(lines), 45):
            chunk = "\n".join(lines[i : i + 45]).strip()
            if len(chunk) > 400:
                code.append((f"{name}:{i + 1}", chunk))
    prose = []
    for name, text in [x for p in _PROSE_FILES for x in _read(p)]:
        for para in text.split("\n\n"):
            para = " ".join(para.split())
            if 400 < len(para) < 2500 and not para.startswith("|"):
                prose.append((name, para))
    if not code or not prose:  # pragma: no cover - repo layout guard
        raise RuntimeError(f"empty corpus under {REPO}")
    return {"code": code, "prose": prose}


# ------------------------------------------------------------- generated facts

_NOUNS = [
    "archive",
    "mission",
    "harbor",
    "beacon",
    "quarry",
    "lattice",
    "ember",
    "pylon",
    "cistern",
    "vellum",
    "orchard",
    "granite",
    "tundra",
    "zephyr",
]
_GREEK = ["KAPPA", "SIGMA", "OMEGA", "TANGO", "DELTA", "LAMBDA", "THETA", "RHO", "IOTA", "ZETA"]
_TEAMS = ["ingest", "ledger", "scheduler", "indexer", "gateway", "replay", "compaction"]
_OWNERS = ["ana", "bo", "cy", "dee", "eli", "fen"]
_REGIONS = ["us-east", "eu-west", "ap-south"]


def _code(rng: random.Random) -> str:
    return f"{rng.randint(1000, 9999)}-{rng.choice(_GREEK)}"


def _table(rng: random.Random, rows: int) -> str:
    """A deterministic structured-fact block: the (c) session family's material."""
    out = []
    for _ in range(rows):
        out.append(
            f"- service {rng.choice(_TEAMS)}-{rng.randint(1, 40)}: owner "
            f"{rng.choice(_OWNERS)}, p99 {rng.randint(12, 900)} ms, "
            f"quota {rng.randint(2, 64)} GiB, region {rng.choice(_REGIONS)}, "
            f"tier {rng.randint(1, 4)}"
        )
    return "\n".join(out)


_SENTENCES = [
    "The owning team reviewed this on the weekly call and left it open.",
    "Nothing here blocks the release; it is recorded for the audit trail only.",
    "A follow-up ticket tracks the remaining cleanup in the compaction path.",
    "This supersedes the note filed two weeks earlier on the same subject.",
    "The measurement was repeated three times and the spread was under five percent.",
    "Operators were notified before the change landed, per the runbook.",
]

_INSTRUCTIONS = [
    ("always write your replies in French", "always write your replies in German"),
    ("always write your replies in all lowercase", "always write your replies in ALL CAPITALS"),
    (
        "always end every reply with the exact marker <<END>>",
        "always end every reply with the exact marker <<STOP>>",
    ),
    (
        "always begin every reply with the exact prefix 'REPORT:'",
        "always begin every reply with the exact prefix 'SUMMARY:'",
    ),
    ("always answer in a single sentence", "always answer as exactly three bullet points"),
]


# ------------------------------------------------------------------- sessions


@dataclass
class Item:
    """One (session, edit) pair plus its queries."""

    sid: int
    family: str
    edit_kind: str
    session: Session
    msg_index: int
    new_content: str
    queries: list = field(default_factory=list)  # (qtype, expected|None, question, forced)
    facts: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _body(rng: random.Random, family: str, corpus: dict) -> tuple[str, str]:
    if family == "code":
        name, chunk = corpus["code"][rng.randrange(len(corpus["code"]))]
        return name, f"Here is {name}:\n```python\n{chunk}\n```"
    if family == "prose":
        name, para = corpus["prose"][rng.randrange(len(corpus["prose"]))]
        return name, f"From {name}: {para}"
    return "table", "Current service inventory:\n" + _table(rng, rng.randint(6, 12))


_USER_ASK = {
    "code": [
        "Walk me through what this does.",
        "Is there a bug in this? Be specific.",
        "How would you test this?",
    ],
    "prose": [
        "Summarise the key claim here.",
        "What is the strongest objection to this?",
        "Rewrite the first sentence more plainly.",
    ],
    "qa": [
        "Which of these is the outlier?",
        "Group these by region.",
        "Which owner carries the most load?",
    ],
}

# tokens a user/assistant pair costs beyond its body: the ask, the reply, the
# planted-fact sentence, the filler sentence, and the chat template's own markup.
_PER_TURN_OVERHEAD = 60

EDIT_KINDS = ["fact", "rewrite", "insert", "delete", "governing"]
FAMILIES = ["code", "prose", "qa"]


def build_item(
    sid: int,
    edit_kind: str,
    family: str,
    corpus: dict,
    seed: int,
    min_tokens: int = 4000,
    max_tokens: int = 8000,
    count_tokens=None,
) -> Item:
    """Build one session (~4-8k tokens), plant three facts, and stage one edit.

    The three planted facts sit strictly before / inside / after the edited message, so
    the same three fact questions are askable under every edit kind. The edited message
    always keeps its fact sentence in its first line; edits other than ``fact`` operate
    around it, never on the code itself.

    ``count_tokens`` maps a string to a token count and sizes the session. Pass the real
    tokenizer: a chars/N estimate overshoots badly on code and on the ``qa`` family's
    short table lines, which is how the first 8B run produced a 10.9k-token session
    against an 8k ceiling. The default is a crude estimate, for CPU tests only.
    """
    count = count_tokens or (lambda s: len(s) // 3)
    rng = random.Random((seed * 1_000_003) ^ (sid * 7919) ^ (EDIT_KINDS.index(edit_kind) * 31))
    instr_a, instr_b = _INSTRUCTIONS[rng.randrange(len(_INSTRUCTIONS))]
    system = (
        "You are a meticulous project assistant reading a long working log. "
        f"Standing instruction for this entire session: {instr_a}. "
        "When asked for a code, reply with the code and nothing else."
    )
    nouns = rng.sample(_NOUNS, 3)
    facts = {k: (nouns[i], _code(rng)) for i, k in enumerate(("before", "at", "after"))}

    session = Session()
    session.turn("system", system)  # governing=True by default for the system role

    # how many user/assistant pairs fit the token budget, measured not estimated.
    # _PER_TURN_OVERHEAD covers the ask, the assistant reply and the template markup
    # that every pair carries on top of its body.
    target = rng.randint(min_tokens, max_tokens)
    bodies: list[tuple[str, str]] = []
    used = count(system) + _PER_TURN_OVERHEAD
    while used < target and len(bodies) < 60:
        body = _body(rng, family, corpus)
        bodies.append(body)
        used += count(body[1]) + _PER_TURN_OVERHEAD
    n_turns = max(6, len(bodies))
    edit_turn = rng.randrange(2, max(3, n_turns - 2))
    before_turn = rng.randrange(0, edit_turn)
    after_turn = rng.randrange(edit_turn + 1, n_turns)

    for t in range(n_turns):
        name, body = bodies[t % len(bodies)]
        planted = ""
        if t == before_turn:
            planted = f" The {facts['before'][0]} code is {facts['before'][1]}."
        if t == edit_turn:
            planted = f" The {facts['at'][0]} code is {facts['at'][1]}."
        if t == after_turn:
            planted = f" The {facts['after'][0]} code is {facts['after'][1]}."
        ask = _USER_ASK[family][t % 3]
        # the filler sentence is what the ``delete`` edit removes, so every user turn
        # carries exactly one and it is never the sentence holding a planted fact
        filler = _SENTENCES[(t + sid) % len(_SENTENCES)]
        session.turn("user", f"Entry {t}.{planted} {filler} {ask}\n{body}")
        session.turn(
            "assistant",
            f"Noted entry {t} ({name}). {rng.choice(_SENTENCES)} "
            f"Team {rng.choice(_TEAMS)} owns the follow-up.",
        )

    msg_index, new_content = _stage_edit(rng, session, edit_kind, edit_turn, facts, family, corpus)
    item = Item(sid, family, edit_kind, session, msg_index, new_content, facts=facts)
    item.meta = {
        "n_turns": n_turns,
        "edit_turn": edit_turn,
        "instruction": instr_a if edit_kind != "governing" else f"{instr_a} -> {instr_b}",
    }
    item.queries = _queries(rng, facts, family, edit_kind)
    return item


def _stage_edit(rng, session, kind, edit_turn, facts, family, corpus):
    """Return (message index, replacement content) for one contiguous single-span edit."""
    if kind == "governing":
        old = session.messages[0]["content"]
        for a, b in _INSTRUCTIONS:
            if a in old:
                return 0, old.replace(a, b)
        raise AssertionError("no standing instruction found")  # pragma: no cover

    idx = 1 + edit_turn * 2  # +1 for the system message; each turn is a user/assistant pair
    old = session.messages[idx]["content"]
    head, body = old.split("\n", 1) if "\n" in old else (old, "")
    if kind == "fact":  # identifier change, delta ~ 0 tokens
        new_code = _code(rng)
        while new_code == facts["at"][1]:
            new_code = _code(rng)
        facts["at_new"] = new_code
        return idx, old.replace(facts["at"][1], new_code)
    if kind == "rewrite":  # body swapped for different material, delta up to +-100 tokens
        _, replacement = _body(rng, family, corpus)
        return idx, head + "\n" + replacement
    if kind == "insert":  # one sentence added inside the edited message
        extra = (
            " "
            + rng.choice(_SENTENCES)
            + " The reviewer added this line after the fact, and it changes nothing else."
        )
        return idx, head + extra + ("\n" + body if body else "")
    if kind == "delete":  # the filler sentence removed from the edited message
        for sentence in _SENTENCES:
            if sentence + " " in head:
                return idx, head.replace(sentence + " ", "", 1) + ("\n" + body if body else "")
        raise AssertionError("no filler sentence to delete")  # pragma: no cover
    raise ValueError(kind)  # pragma: no cover


def _queries(rng, facts, family, edit_kind):
    """Query pool: fact before / at / after the edit, summarise, obey, continue code."""
    at_code = facts.get("at_new", facts["at"][1])
    fact_at = (
        "fact-at",
        [at_code],
        f"What is the {facts['at'][0]} code?",
        f" The {facts['at'][0]} code is",
    )
    obey = ("obey", None, "Give me a one-line status of this project.", "")
    pool = [
        (
            "fact-before",
            [facts["before"][1]],
            f"What is the {facts['before'][0]} code?",
            f" The {facts['before'][0]} code is",
        ),
        (
            "fact-after",
            [facts["after"][1]],
            f"What is the {facts['after'][0]} code?",
            f" The {facts['after'][0]} code is",
        ),
        ("summarise", None, "Summarise the most recent change described in the log above.", ""),
        obey,
    ]
    if family == "code":
        pool.append(
            ("continue-code", None, "Continue the last Python snippet above with a few lines.", "")
        )
    rng.shuffle(pool)
    picks = [fact_at, *pool[:2]]
    if edit_kind == "governing" and not any(p[0] == "obey" for p in picks):
        picks[-1] = obey  # a governing edit is only testable if something must obey it
    return picks


# ---------------------------------------------------------------- rendering


def render(tok, messages, add_generation_prompt=False) -> str:
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def question_text(tok, messages, question: str, forced_prefix: str) -> str:
    """The final user turn, rendered as the template renders a user turn.

    Qwen3's template is not append-only — it injects the empty ``<think></think>`` block
    into whichever assistant message is *last*, so appending a user turn rewrites the
    turn before it. The suffix is therefore derived against the system message alone
    (the same trick ``kvshift_probe`` uses); the history keeps the think block on its
    final assistant turn, identically under every condition.
    """
    stem = messages[:1]
    head = render(tok, stem)
    full = render(tok, [*stem, {"role": "user", "content": question}], True)
    assert full.startswith(head), "chat template is not append-only even for one turn"
    return full[len(head) :] + forced_prefix


# ------------------------------------------------------------------ reporting


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:  # pragma: no cover
        return float("nan")
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarise(rows: list[dict], key: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r[key], r["condition"]), []).append(r)
    out = []
    for (bucket, cond), rs in sorted(groups.items()):
        kl = [r["kl_mean_forced"] for r in rs]
        out.append(
            {
                "bucket": bucket,
                "condition": cond,
                "n": len(rs),
                "kl_mean": statistics.fmean(kl),
                "kl_median": statistics.median(kl),
                "kl_p95": _pct(kl, 0.95),
                "kl_max": max(kl),
                "kl_first_mean": statistics.fmean([r["kl_first"] for r in rs]),
                "kl_first_max": max(r["kl_first"] for r in rs),
                "tf_top1": statistics.fmean([r["tf_top1_agree"] for r in rs]),
                "exact": statistics.fmean([float(r["exact_match"]) for r in rs]),
                "frac": statistics.fmean([r["recompute_frac"] for r in rs]),
                "over_005": sum(k > 0.05 for k in kl),
                "over_02": sum(k > 0.2 for k in kl),
            }
        )
    return out


_HDR = (
    f"{'bucket':<14}{'condition':<16}{'n':>4}{'klmean':>9}{'klmed':>9}{'klp95':>9}"
    f"{'klmax':>9}{'kl1mean':>9}{'kl1max':>9}{'tf_top1':>9}{'exact':>7}{'frac':>7}"
    f"{'>.05':>6}{'>.2':>5}"
)


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n### {title}\n{_HDR}")
    for r in rows:
        print(
            f"{str(r['bucket']):<14}{r['condition']:<16}{r['n']:>4}{r['kl_mean']:>9.4f}"
            f"{r['kl_median']:>9.4f}{r['kl_p95']:>9.4f}{r['kl_max']:>9.4f}"
            f"{r['kl_first_mean']:>9.4f}{r['kl_first_max']:>9.4f}{r['tf_top1']:>9.3f}"
            f"{r['exact']:>7.2f}{r['frac']:>7.3f}{r['over_005']:>6}{r['over_02']:>5}"
        )


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="marathon.kvshift_eval", description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--sessions", type=int, default=60, help="number of (session, edit) items")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--min-tokens", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument(
        "--model-question-frac",
        type=float,
        default=0.2,
        help="fraction of items given an extra query the model writes itself",
    )
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            attn_implementation=args.attn,
        )
        .to(args.device)
        .eval()
    )
    dev = next(model.parameters()).device

    def ids(text):
        return torch.tensor(tok.encode(text, add_special_tokens=False), device=dev)

    corpus = load_corpus()
    plan_rng = random.Random(args.seed)

    items = [
        build_item(
            sid,
            EDIT_KINDS[sid % len(EDIT_KINDS)],
            FAMILIES[(sid // len(EDIT_KINDS)) % len(FAMILIES)],
            corpus,
            args.seed,
            args.min_tokens,
            args.max_tokens,
            lambda s: len(tok.encode(s, add_special_tokens=False)),
        )
        for sid in range(args.sessions)
    ]

    policies = [
        Policy("none"),
        Policy("none", rerotate=False),
        Policy("firstm", m=10**9, name="prefix-equiv"),
    ]
    rows: list[dict] = []
    jsonl = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None  # noqa: SIM115
    t_start = time.perf_counter()
    lengths = []

    for n_done, item in enumerate(items, 1):
        old_text = render(tok, item.session.messages)
        item.session.edit(item.msg_index, item.new_content)
        new_text = render(tok, item.session.messages)
        head, tail_b = byte_span(old_text.encode(), new_text.encode())
        old_ids, new_ids = ids(old_text), ids(new_text)
        span = token_span(old_ids.tolist(), new_ids.tolist())
        lengths.append(int(new_ids.shape[0]))
        old_kv, _ = prefill(model, old_ids)

        queries = list(item.queries)
        if plan_rng.random() < args.model_question_frac:
            ask = "Write one short factual question about the log above. Output the question only."
            q0 = ids(question_text(tok, item.session.messages, ask, ""))
            written = tok.decode(run_full(model, new_ids, q0, 24)["tokens"]).strip().strip('"')
            written = written.split("?")[0].strip() + "?"
            if 10 < len(written) < 200:
                queries.append(("model-written", None, written, ""))

        start_row = len(rows)
        for qtype, expected, question, forced_prefix in queries:
            q = ids(question_text(tok, item.session.messages, question, forced_prefix))
            ref = run_full(model, new_ids, q, args.gen_tokens)
            ref_text = tok.decode(ref["tokens"])
            base = {
                "sid": item.sid,
                "family": item.family,
                "edit_kind": item.edit_kind,
                "qtype": qtype,
                "question": question,
                "span_p": span.p,
                "span_e_old": span.e_old,
                "span_e_new": span.e_new,
                "span_s": span.s,
                "delta": span.delta,
                "byte_head": head,
                "byte_tail": tail_b,
                "prompt_tokens": int(new_ids.shape[0] + q.shape[0]),
            }
            rows.append(
                {
                    **base,
                    "condition": "full-recompute",
                    "recompute_frac": 1.0,
                    "effective_frac": 1.0,
                    "prefill_s": ref["prefill_s"],
                    "kl_first": 0.0,
                    "kl_mean_forced": 0.0,
                    "kl_max_forced": 0.0,
                    "tf_top1_agree": 1.0,
                    "exact_match": True,
                    "answer_ok": expected is None or any(e in ref_text for e in expected),
                    "text": ref_text,
                }
            )
            for pol in policies:
                got = run_policy(
                    model, old_kv, span, new_ids, q, pol, args.gen_tokens, forced=ref["tokens"]
                )
                text = tok.decode(got["tokens"])
                cmp = compare(ref, got)
                rows.append(
                    {
                        **base,
                        "condition": pol.label(),
                        "recompute_frac": got["recompute_frac"],
                        "effective_frac": got["effective_frac"],
                        "prefill_s": got["prefill_s"],
                        "kl_first": cmp["kl_first"],
                        "kl_mean_forced": cmp["kl_mean_forced"],
                        "kl_max_forced": cmp["kl_max_forced"],
                        "tf_top1_agree": cmp["tf_top1_agree"],
                        "exact_match": got["tokens"] == ref["tokens"],
                        "answer_ok": expected is None or any(e in text for e in expected),
                        "text": text,
                    }
                )
        if jsonl:
            for r in rows[start_row:]:
                jsonl.write(json.dumps(r) + "\n")
            jsonl.flush()
        del old_kv
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[{n_done}/{len(items)}] {item.edit_kind:<10}{item.family:<6}"
            f"len={lengths[-1]} P={span.p} E={span.e_old}->{span.e_new} "
            f"d={span.delta} S={span.s} ({time.perf_counter() - t_start:.0f}s)",
            flush=True,
        )

    if jsonl:
        jsonl.close()
    print(
        f"\nhistory tokens: min={min(lengths)} median={statistics.median(lengths):.0f} "
        f"max={max(lengths)} over {len(items)} sessions"
    )
    tables = [
        ("overall", summarise([{**r, "all": "ALL"} for r in rows], "all")),
        ("by edit kind", summarise(rows, "edit_kind")),
        ("by query type", summarise(rows, "qtype")),
        ("by session family", summarise(rows, "family")),
    ]
    for title, table in tables:
        print_table(title, table)

    reuse = [r for r in rows if r["condition"] == "reuse-all"]
    gov = [r for r in reuse if r["edit_kind"] == "governing"]
    non = [r for r in reuse if r["edit_kind"] != "governing"]
    print(
        f"\nreuse-all items over KL 0.2: {sum(r['kl_mean_forced'] > 0.2 for r in reuse)}"
        f"/{len(reuse)} (governing {sum(r['kl_mean_forced'] > 0.2 for r in gov)}/{len(gov)}, "
        f"non-governing {sum(r['kl_mean_forced'] > 0.2 for r in non)}/{len(non)})"
    )
    print(
        f"reuse-all exact-match vs reference: governing "
        f"{statistics.fmean([float(r['exact_match']) for r in gov]):.2f}, non-governing "
        f"{statistics.fmean([float(r['exact_match']) for r in non]):.2f}"
    )
    fact_reuse = [r for r in reuse if r["qtype"].startswith("fact")]
    fact_ref = [
        r for r in rows if r["condition"] == "full-recompute" and r["qtype"].startswith("fact")
    ]
    print(
        f"planted-fact answers correct: reuse-all "
        f"{sum(r['answer_ok'] for r in fact_reuse)}/{len(fact_reuse)}, full-recompute "
        f"{sum(r['answer_ok'] for r in fact_ref)}/{len(fact_ref)}"
    )

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "seed": args.seed,
                    "n_items": len(items),
                    "history_tokens": lengths,
                    **{title.replace(" ", "_"): table for title, table in tables},
                },
                f,
                indent=1,
            )
    if dev.type == "cuda":
        print(f"\npeak GPU MiB: {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    print(f"total {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
