"""Probe: quality vs recompute-fraction for position-shifted KV reuse (see kvshift.py).

Builds a real session with :class:`marathon.session.Session`, edits one turn in place,
lets the delta engine locate the changed span, then answers three planted-fact
questions from a *stitched* cache (P reused, E' fresh, S re-rotated + partially
recomputed) and compares against a full recompute of the same new sequence.

    python -m marathon.kvshift_probe --model Qwen/Qwen3-0.6B --turns 20

Not run in CI (needs a GPU and weights). Unit tests cover the pure functions.
"""

from __future__ import annotations

import argparse
import json
import time

from .kvshift import (
    Policy,
    byte_span,
    compare,
    inv_freq_of,
    prefill,
    rerotate_keys,
    run_full,
    run_policy,
    token_span,
)
from .session import Session

_SYSTEM = (
    "You are a careful assistant reading a long project log. "
    "The archive code is 7391-KAPPA. Answer questions with the exact code asked for."
)

_TOPICS = [
    "The build pipeline was reorganised so that artefacts are content addressed",
    "Latency on the ingest path fell after the batching window was widened",
    "A migration moved the ledger snapshots onto append-only storage",
    "The scheduler now backs off exponentially when the queue drains slowly",
    "Documentation for the delta wire format was rewritten from scratch",
    "Two flaky integration tests were traced to a clock skew in the runner",
    "Memory use in the indexer dropped once the rolling checksum was reused",
    "The retention policy for cold segments was shortened to thirty days",
    "A regression in the tokenizer cache was found by the replay gate",
    "Operators asked for per-session metrics to be exported, not printed",
]


def _paragraph(turn: int, repeat: int) -> str:
    """Deterministic but varied prose, ~220 tokens per turn."""
    parts = []
    for j in range(repeat):
        topic = _TOPICS[(turn + j) % len(_TOPICS)]
        parts.append(
            f"{topic}; this was reviewed on day {turn * 7 + j} and the owning team "
            f"recorded {40 + (turn * 3 + j) % 17} open items with a median age of "
            f"{2 + (turn + j) % 9} days."
        )
    return " ".join(parts)


def build_session(turns: int, edit_turn: int, fact_gap: int) -> tuple[Session, dict]:
    """20-ish turns of varied content with three unique planted facts."""
    facts = {
        "prefix": ("archive", "7391-KAPPA"),  # lives in the system prompt (always P)
        "edit": ("mission", "5520-DELTA"),  # lives in the edited turn
        "suffix": ("harbor", "8814-OMEGA"),  # lives after the edit (S)
    }
    session = Session()
    s_turn = min(edit_turn + fact_gap, turns - 1)
    for t in range(turns):
        extra = ""
        if t == edit_turn:
            extra = f" The {facts['edit'][0]} code is {facts['edit'][1]}."
        if t == s_turn:
            extra = f" The {facts['suffix'][0]} code is {facts['suffix'][1]}."
        session.turn("user", f"Log entry {t}.{extra} {_paragraph(t, 6)}")
        session.turn("assistant", f"Noted entry {t}.")
    return session, facts


def render(session: Session) -> str:
    return _SYSTEM + "\n" + "\n".join(
        f"{m['role']}: {m['content']}" for m in session.messages
    )


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="marathon.kvshift_probe", description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--open-tokens", type=int, default=48,
                    help="greedy tokens for the open-ended (unforced) question")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenario", default=None, help="run just this scenario")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        attn_implementation=args.attn,
    ).to(args.device).eval()
    dev = next(model.parameters()).device

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def ids(text: str):
        return torch.tensor(tok.encode(text, add_special_tokens=False), device=dev)

    # --- rerotation self-check on the real model's inv_freq -------------------
    inv = inv_freq_of(model)
    hd = inv.shape[0] * 2
    k0 = torch.randn(1, 4, 6, hd, device=dev, dtype=torch.float32)
    from .kvshift import rotate_half

    def rope(k, p):
        ang = torch.arange(p, p + k.shape[2], device=dev, dtype=torch.float32)[:, None] * inv
        emb = torch.cat((ang, ang), dim=-1)
        return k * emb.cos() + rotate_half(k) * emb.sin()

    err = (rerotate_keys(rope(k0, 100), 37, inv) - rope(k0, 137)).abs().max().item()
    print(f"rerotate max abs error (fp32, real inv_freq): {err:.3e}")

    scenarios = [
        ("edit-turn0", 0, 5, "[EDITED] ", None),
        ("edit-mid", 10, 3, "[EDITED] ", None),
        ("edit-grow", 10, 3, "[EDITED] ", 50),
    ]
    policies = [
        Policy("none", rerotate=False),  # control: reuse S's keys unrotated
        Policy("none"),
        Policy("firstm", m=32),
        Policy("firstm", m=128),
        Policy("blend", ratio=0.05),
        Policy("blend", ratio=0.15),
        Policy("blend", ratio=0.30),
    ]
    report: list[dict] = []

    if args.scenario:
        scenarios = [s for s in scenarios if s[0] == args.scenario]
    for name, edit_turn, gap, marker, grow in scenarios:
        session, facts = build_session(args.turns, edit_turn, gap)
        msg_index = edit_turn * 2  # user messages are at even indices
        old_text = render(session)
        old_content = session.messages[msg_index]["content"]
        new_content = marker + old_content.replace(facts["edit"][1], "9902-SIGMA")
        if grow:
            new_content += " " + _paragraph(99, grow // 12 + 1)
        session.edit(msg_index, new_content)
        new_text = render(session)

        head, tail_b = byte_span(old_text.encode(), new_text.encode())
        old_ids, new_ids = ids(old_text), ids(new_text)
        span = token_span(old_ids.tolist(), new_ids.tolist())
        print(
            f"\n== {name}: byte delta head={head} tail={tail_b} | tokens "
            f"P={span.p} E={span.e_old}->{span.e_new} (d={span.delta}) S={span.s}"
        )

        t0 = time.perf_counter()
        old_kv, _ = prefill(model, old_ids)
        _sync()
        print(f"   old-sequence prefill ({old_ids.shape[0]} tok): {time.perf_counter()-t0:.3f}s")

        questions = []
        for which, (fact_name, code) in facts.items():
            code = "9902-SIGMA" if which == "edit" else code
            questions.append(
                (
                    which,
                    code,
                    f"\nuser: What is the {fact_name} code?"
                    f"\nassistant: The {fact_name} code is",
                    args.max_new_tokens,
                )
            )
        # open-ended: nothing forces the answer, so divergence shows up here first
        questions.append(
            ("open", None, "\nuser: Summarise the log so far.\nassistant:", args.open_tokens)
        )

        for which, code, prompt, n_tok in questions:
            q = ids(prompt)
            ref = run_full(model, new_ids, q, n_tok)
            ref_text = tok.decode(ref["tokens"])
            rows = [
                {
                    **{k: ref[k] for k in ("policy", "recomputed_tokens", "recompute_frac",
                                           "effective_frac", "prefill_s", "wall_s")},
                    "text": ref_text,
                    "exact": code is None or code in ref_text,
                    "kl_first": 0.0,
                    "kl_mean_forced": 0.0,
                    "kl_max_forced": 0.0,
                    "tf_top1_agree": 1.0,
                    "greedy_prefix_agree": 1.0,
                    "top1_match": True,
                    "max_logit_diff": 0.0,
                }
            ]
            for pol in policies:
                got = run_policy(
                    model, old_kv, span, new_ids, q, pol, n_tok, forced=ref["tokens"]
                )
                text = tok.decode(got["tokens"])
                rows.append(
                    {
                        **{k: got[k] for k in ("policy", "recomputed_tokens", "recompute_frac",
                                               "effective_frac", "prefill_s", "wall_s")},
                        "text": text,
                        "exact": code is None or code in text,
                        **compare(ref, got),
                    }
                )
            print(f"  -- question: {which} (expects {code})")
            print(f"     {'policy':<16}{'frac':>7}{'eff':>7}{'prefill_s':>10}{'kl1':>9}"
                  f"{'klmean':>9}{'klmax':>9}{'tf_top1':>9}{'agree':>7}{'exact':>7}  text")
            for r in rows:
                print(
                    f"     {r['policy']:<16}{r['recompute_frac']:>7.3f}"
                    f"{r['effective_frac']:>7.3f}{r['prefill_s']:>10.3f}{r['kl_first']:>9.4f}"
                    f"{r['kl_mean_forced']:>9.4f}{r['kl_max_forced']:>9.4f}"
                    f"{r['tf_top1_agree']:>9.2f}{r['greedy_prefix_agree']:>7.2f}"
                    f"{str(r['exact']):>7}  {r['text']!r}"
                )
                report.append(
                    {"scenario": name, "question": which,
                     **{k: v for k, v in r.items() if k not in ("logits", "logits_seq")}}
                )
        del old_kv
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "attn": args.attn, "rerotate_err": err,
                       "rows": report}, f, indent=1)
    if dev.type == "cuda":
        print(f"\npeak GPU MiB: {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
