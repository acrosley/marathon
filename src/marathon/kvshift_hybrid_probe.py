"""Probe: does delta-driven reuse survive on a Gated-DeltaNet hybrid? (see kvshift_hybrid.py)

Same sessions, same scenarios, same metrics as ``kvshift_probe`` — so the numbers are
directly comparable to the dense Qwen3 runs — but the model is a hybrid whose linear
layers have no KV, and the policies are the hybrid ones.

    python -m marathon.kvshift_hybrid_probe --model Qwen/Qwen3.5-4B --turns 20

Not run in CI (needs a GPU and weights).
"""

from __future__ import annotations

import argparse
import json
import time

from .kvshift import Span, byte_span, compare, token_span
from .kvshift_hybrid import (
    CostModel,
    HybridPolicy,
    capture_old,
    check_rope,
    run_full_hybrid,
    run_hybrid,
)
from .kvshift_probe import _template, build_edit

_ANCHOR = "<|im_start|>user"


def _split(tok, session, question: str) -> tuple[str, str]:
    """Render the session and one question turn, split at the question's own header.

    ``kvshift_probe.question_text`` assumes the template is append-only. Qwen3.5's is
    not: it injects an empty ``<think></think>`` block into whichever assistant turn
    is *last*, so adding a user turn rewrites the message before it. Rendering the
    question turn as part of the prompt and cutting at its header sidesteps that, and
    keeps the context render identical across questions (the cut point does not
    depend on what the question says).
    """
    full = _template(tok, [*session.messages, {"role": "user", "content": question}], True)
    return full[: full.rindex(_ANCHOR)], full[full.rindex(_ANCHOR) :]


def render_context(tok, session) -> str:
    """The session as the model will see it, with a question turn about to follow."""
    return _split(tok, session, "?")[0]


def question_suffix(tok, session, question: str, forced_prefix: str) -> str:
    return _split(tok, session, question)[1] + forced_prefix


SCENARIOS = {
    "edit-turn0": lambda t: build_edit(t, 0, 5, None),
    "edit-mid": lambda t: build_edit(t, 10, 3, None),
    "edit-grow": lambda t: build_edit(t, 10, 3, 200),
}


def policies(first_m: int) -> list[HybridPolicy]:
    return [
        HybridPolicy(linear="stale", rerotate=False),  # control: keys at stale angles
        HybridPolicy(linear="stale"),
        HybridPolicy(linear="mix"),
        HybridPolicy(linear="mix", first_m=first_m),
        HybridPolicy(linear="hidden"),
        HybridPolicy(linear="hidden", first_m=first_m),
    ]


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="marathon.kvshift_hybrid_probe", description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--first-m", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenario", default=None, help="comma-separated scenarios to run")
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
    inner = model.model
    kinds = list(inner.config.layer_types)
    cost = CostModel.of(inner)
    print(f"model {args.model}: {len(kinds)} layers, {cost.n_linear} linear / {cost.n_attn} full")
    print(
        f"cost model (params/token): full={cost.full_token / 1e6:.1f}M  "
        f"replay-hidden={cost.replay_hidden / 1e6:.1f}M "
        f"({cost.replay_hidden / cost.full_token:.3f} of full)  "
        f"replay-mix={cost.replay_mix / 1e6:.1f}M ({cost.replay_mix / cost.full_token:.4f})"
    )
    print(f"mRoPE re-rotation max abs error on the real module: {check_rope(inner, dev):.3e}")

    def ids(text: str):
        return torch.tensor(tok.encode(text, add_special_tokens=False), device=dev)

    report: list[dict] = []
    names = args.scenario.split(",") if args.scenario else list(SCENARIOS)
    for name in names:
        session, mutate, questions = SCENARIOS[name](args.turns)
        old_text = render_context(tok, session)
        mutate(session)
        new_text = render_context(tok, session)
        head, tail_b = byte_span(old_text.encode(), new_text.encode())
        old_ids, new_ids = ids(old_text), ids(new_text)
        span: Span = token_span(old_ids.tolist(), new_ids.tolist())
        print(
            f"\n== {name}: byte delta head={head} tail={tail_b} | "
            f"P={span.p} E={span.e_old}->{span.e_new} (d={span.delta}) S={span.s} "
            f"new_len={span.new_len}"
        )

        t0 = time.perf_counter()
        old = capture_old(model, old_ids, span)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        mem = old.bytes()
        print(
            f"   old-turn capture ({old_ids.shape[0]} tok): {time.perf_counter() - t0:.2f}s  "
            f"kv={mem['kv_mib']:.0f} MiB  s_hidden={mem['s_hidden_mib']:.0f} MiB  "
            f"s_mix={mem['s_mix_mib']:.0f} MiB"
        )

        for which, expected, question, forced_prefix, n_tok in questions:
            q = ids(question_suffix(tok, session, question, forced_prefix))
            ref = run_full_hybrid(model, new_ids, q, n_tok)
            ref_text = tok.decode(ref["tokens"])
            rows = [
                {
                    **{
                        k: ref[k]
                        for k in ("policy", "fresh_frac", "flop_frac", "prefill_s", "wall_s")
                    },
                    "text": ref_text,
                    "exact": expected is None or any(e in ref_text for e in expected),
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
            for pol in policies(args.first_m):
                got = run_hybrid(model, old, span, new_ids, q, pol, n_tok, forced=ref["tokens"])
                text = tok.decode(got["tokens"])
                rows.append(
                    {
                        **{
                            k: got[k]
                            for k in ("policy", "fresh_frac", "flop_frac", "prefill_s", "wall_s")
                        },
                        "text": text,
                        "exact": expected is None or any(e in text for e in expected),
                        "same_as_ref": text.strip() == ref_text.strip(),
                        **compare(ref, got),
                    }
                )
            print(f"  -- question: {which} (expects {expected}, full recompute said {ref_text!r})")
            print(
                f"     {'policy':<22}{'fresh':>7}{'flops':>8}{'prefill_s':>10}{'wall_s':>9}"
                f"{'klmean':>9}{'klmax':>9}{'tf_top1':>9}{'exact':>7}{'==ref':>7}  text"
            )
            for r in rows:
                print(
                    f"     {r['policy']:<22}{r['fresh_frac']:>7.3f}{r['flop_frac']:>8.3f}"
                    f"{r['prefill_s']:>10.3f}{r['wall_s']:>9.3f}{r['kl_mean_forced']:>9.4f}"
                    f"{r['kl_max_forced']:>9.4f}{r['tf_top1_agree']:>9.2f}"
                    f"{str(r['exact']):>7}{str(r['same_as_ref']):>7}  {r['text']!r}"
                )
                report.append(
                    {
                        "scenario": name,
                        "question": which,
                        **{k: v for k, v in r.items() if k not in ("logits", "logits_seq")},
                    }
                )
        del old
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "layer_types": kinds, "rows": report}, f, indent=1)
    if dev.type == "cuda":
        print(f"\npeak GPU MiB: {torch.cuda.max_memory_allocated() / 2**20:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
