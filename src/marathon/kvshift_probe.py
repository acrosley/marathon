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


# --- dependent-span scenarios --------------------------------------------
# In these, S does not merely sit after the edit: its meaning is a function of
# the edited span, so re-rotation alone (which fixes positions, not content)
# should not be enough. What counts as right is the *full recompute of the same
# new sequence* -- that is ground truth here, not our intent.

_OLD_CODE, _NEW_CODE = "5520-DELTA", "9902-SIGMA"
_FR = {
    "le",
    "la",
    "les",
    "de",
    "des",
    "du",
    "et",
    "est",
    "pour",
    "dans",
    "une",
    "un",
    "sur",
    "avec",
    "ete",
    "qui",
    "plus",
    "cette",
    "sont",
    "nous",
}
_DE = {
    "der",
    "die",
    "das",
    "und",
    "ist",
    "mit",
    "auf",
    "den",
    "dem",
    "ein",
    "eine",
    "wurde",
    "werden",
    "nicht",
    "von",
    "im",
    "sich",
    "auch",
    "wir",
}


def _lang(text: str) -> str:
    words = {w.strip(".,;:!?()").lower() for w in text.split()}
    fr, de = len(words & _FR), len(words & _DE)
    return "de" if de > fr else "fr" if fr > de else "??"


def _log(session, turns, extras):
    for t in range(turns):
        session.turn("user", f"Log entry {t}.{extras.get(t, '')} {_paragraph(t, 6)}")
        session.turn("assistant", f"Noted entry {t}.")


def build_dep_anaphora(turns: int):
    """S refers back to the edited value *without restating it* ("that code")."""
    src = 2
    session = Session()
    _log(
        session,
        turns,
        {
            src: f" The mission code is {_OLD_CODE}.",
            src + 3: (
                " As stated in the earlier entry above, that mission code is from now on "
                "the primary key for this project; whenever anyone asks for the primary "
                "key, answer with the mission code given in that entry."
            ),
            src + 6: (
                " Reminder: the primary key is exactly the mission code given above, "
                "copied verbatim, and no other value may be substituted for it."
            ),
        },
    )
    new = session.messages[src * 2]["content"].replace(_OLD_CODE, _NEW_CODE)
    qs = [
        ("primary-key", [_NEW_CODE], "What is the primary key?", " The primary key is", 12),
        ("mission", [_NEW_CODE], "What is the mission code?", " The mission code is", 12),
        ("open", None, "Summarise the log so far.", "", 48),
    ]
    return session, src * 2, new, qs


def build_dep_instruction(turns: int):
    """Turn 0 flips a standing instruction that governs the *final* answer."""
    session = Session()
    _log(
        session,
        turns,
        {
            0: (
                " Standing instruction for this entire session: always write your replies "
                "in French, whatever the language of the question."
            ),
        },
    )
    new = session.messages[0]["content"].replace("in French", "in German")
    qs = [
        ("lang-pipeline", ["de"], "Describe the build pipeline in one sentence.", "", 40),
        ("lang-scheduler", ["de"], "What changed about the scheduler?", "", 40),
    ]
    return session, 0, new, qs


def build_dep_contradict(turns: int):
    """A new constraint is inserted mid-history that later text contradicts."""
    src = 10
    session = Session()
    _log(
        session,
        turns,
        {
            src + 3: (
                " The harbor code is 8814-OMEGA; quote it verbatim whenever anyone "
                "asks for the harbor code."
            ),
        },
    )
    new = session.messages[src * 2]["content"] + (
        " Correction, authoritative and overriding every later mention in this log: "
        "the harbor code 8814-OMEGA has been revoked and is invalid; it must never be "
        "quoted again. The only valid harbor code is 4417-TANGO."
    )
    qs = [
        (
            "harbor",
            ["4417-TANGO"],
            "What is the valid harbor code?",
            " The valid harbor code is",
            12,
        ),
        ("open", None, "\nuser: Summarise the log so far.\nassistant:", 48),
    ]
    return session, src * 2, new, qs


def build_edit(turns: int, edit_turn: int, gap: int, grow):
    """The original independent-S scenarios, in the shared builder shape."""
    session, facts = build_session(turns, edit_turn, gap)
    idx = edit_turn * 2
    new = "[EDITED] " + session.messages[idx]["content"].replace(facts["edit"][1], _NEW_CODE)
    if grow:
        new += " " + _paragraph(99, grow // 12 + 1)
    qs = [
        (
            which,
            [_NEW_CODE if which == "edit" else code],
            f"What is the {fact} code?",
            f" The {fact} code is",
            12,
        )
        for which, (fact, code) in facts.items()
    ]
    qs.append(("open", None, "Summarise the log so far.", "", 48))
    return session, idx, new, qs


def render(session: Session, tok=None) -> str:
    """Plain transcript, or the model's own chat template when `tok` is given.

    Without a template a base model just *continues the log* instead of obeying
    it, which silently turns every instruction-following test into a no-op.
    """
    if tok is None:
        return _SYSTEM + "\n" + "\n".join(f"{m['role']}: {m['content']}" for m in session.messages)
    return _template(tok, session.messages)


def _template(tok, messages, add_generation_prompt=False) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": _SYSTEM}, *messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def question_text(tok, question: str, forced_prefix: str) -> str:
    """The question as a fresh user turn, in whatever format the model expects."""
    if tok is None:
        return f"\nuser: {question}\nassistant:{forced_prefix}"
    head = _template(tok, [])
    full = _template(tok, [{"role": "user", "content": question}], True)
    assert full.startswith(head)
    return full[len(head) :] + forced_prefix


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="marathon.kvshift_probe", description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument(
        "--open-tokens",
        type=int,
        default=48,
        help="greedy tokens for the open-ended (unforced) question",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenario", default=None, help="run just this scenario")
    ap.add_argument(
        "--raw", action="store_true", help="plain transcript instead of the model's chat template"
    )
    ap.add_argument("--json", default=None)
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
        # S is semantically independent of the edit (original three)
        ("edit-turn0", lambda t: build_edit(t, 0, 5, None)),
        ("edit-mid", lambda t: build_edit(t, 10, 3, None)),
        ("edit-grow", lambda t: build_edit(t, 10, 3, 50)),
        # S semantically depends on the edited span
        ("dep-anaphora", build_dep_anaphora),
        ("dep-instruction", build_dep_instruction),
        ("dep-contradict", build_dep_contradict),
    ]
    policies = [
        Policy("none", rerotate=False),  # control: reuse S's keys unrotated
        Policy("none"),
        Policy("firstm", m=32),
        Policy("firstm", m=128),
        Policy("firstm", m=512),
        Policy("blend", ratio=0.05),
        Policy("blend", ratio=0.15),
        Policy("blend", ratio=0.30),
    ]
    report: list[dict] = []

    def label(which: str, text: str) -> str:
        """What we grade on: the language for instruction-following, else the text."""
        return _lang(text) if which.startswith("lang") else text.strip()

    if args.scenario:
        scenarios = [sc for sc in scenarios if sc[0] == args.scenario]
    for name, builder in scenarios:
        session, msg_index, new_content, questions = builder(args.turns)
        chat_tok = None if args.raw else tok
        old_text = render(session, chat_tok)
        session.edit(msg_index, new_content)
        new_text = render(session, chat_tok)

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
        print(f"   old-sequence prefill ({old_ids.shape[0]} tok): {time.perf_counter() - t0:.3f}s")

        for which, expected, question, forced_prefix, n_tok in questions:
            q = ids(question_text(chat_tok, question, forced_prefix))
            ref = run_full(model, new_ids, q, n_tok)
            ref_text = tok.decode(ref["tokens"])
            ref_label = label(which, ref_text)

            def graded(text, which=which, expected=expected, ref_label=ref_label):
                """(matches what we intended, matches the full-recompute answer)."""
                lab = label(which, text)
                ok = expected is None or any(e in lab for e in expected)
                return ok, lab == ref_label

            rows = [
                {
                    **{
                        k: ref[k]
                        for k in (
                            "policy",
                            "recomputed_tokens",
                            "recompute_frac",
                            "effective_frac",
                            "prefill_s",
                            "wall_s",
                        )
                    },
                    "text": ref_text,
                    "exact": graded(ref_text)[0],
                    "same_as_ref": True,
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
                got = run_policy(model, old_kv, span, new_ids, q, pol, n_tok, forced=ref["tokens"])
                text = tok.decode(got["tokens"])
                ok, same = graded(text)
                rows.append(
                    {
                        **{
                            k: got[k]
                            for k in (
                                "policy",
                                "recomputed_tokens",
                                "recompute_frac",
                                "effective_frac",
                                "prefill_s",
                                "wall_s",
                            )
                        },
                        "text": text,
                        "exact": ok,
                        "same_as_ref": same,
                        **compare(ref, got),
                    }
                )
            print(
                f"  -- question: {which} (expects {expected}, full recompute said "
                f"{ref_label[:44]!r})"
            )
            print(
                f"     {'policy':<16}{'frac':>7}{'eff':>7}{'prefill_s':>10}{'kl1':>9}"
                f"{'klmean':>9}{'klmax':>9}{'tf_top1':>9}{'agree':>7}{'exact':>7}"
                f"{'==ref':>7}  text"
            )
            for r in rows:
                print(
                    f"     {r['policy']:<16}{r['recompute_frac']:>7.3f}"
                    f"{r['effective_frac']:>7.3f}{r['prefill_s']:>10.3f}{r['kl_first']:>9.4f}"
                    f"{r['kl_mean_forced']:>9.4f}{r['kl_max_forced']:>9.4f}"
                    f"{r['tf_top1_agree']:>9.2f}{r['greedy_prefix_agree']:>7.2f}"
                    f"{str(r['exact']):>7}{str(r['same_as_ref']):>7}  {r['text']!r}"
                )
                report.append(
                    {
                        "scenario": name,
                        "question": which,
                        **{k: v for k, v in r.items() if k not in ("logits", "logits_seq")},
                    }
                )
        del old_kv
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {"model": args.model, "attn": args.attn, "rerotate_err": err, "rows": report},
                f,
                indent=1,
            )
    if dev.type == "cuda":
        print(f"\npeak GPU MiB: {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
