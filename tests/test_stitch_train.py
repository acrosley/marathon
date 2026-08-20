"""CPU cover for the stitched-KV fine-tuning plumbing.

No weights and no GPU: a randomly initialised 2-layer Qwen3 and a byte-level tokenizer
stand in for the real thing, so what is under test is the wiring — LoRA wrapping and the
teacher/student toggle, the differentiable stitch, the KL loss, and the fact that the
gradient actually reaches the adapter through a stitched cache. The numbers in
findings.md come from the GPU runs; these tests only keep the harness honest.
"""

from __future__ import annotations

import json
import random
import statistics

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from marathon.kvshift import span_segments, token_span  # noqa: E402
from marathon.stitch_train import (  # noqa: E402
    Example,
    GradShiftCache,
    LoRALinear,
    _prefill,
    adapters,
    apply_lora,
    build_examples,
    clean_logits,
    evaluate,
    example_losses,
    grad_stitch,
    kl_to,
    load_lora_state,
    lora_state,
    report,
    stitched_logits,
    teacher_reference,
    train,
)

VOCAB = 260


class ByteTokenizer:
    """Enough of the tokenizer API for the builders: bytes in, ids out."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8")[:200_000]) or [0]

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return out + ("<|assistant|>" if add_generation_prompt else "")


@pytest.fixture
def tiny():
    from transformers import AutoModelForCausalLM, Qwen3Config

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()
    return model, apply_lora(model, r=4, alpha=8)


def _example(device="cpu", n=48, q=6):
    rng = random.Random(3)
    old = [rng.randrange(1, VOCAB) for _ in range(n)]
    new = old[:10] + [rng.randrange(1, VOCAB) for _ in range(4)] + old[12:]
    span = token_span(old, new)
    return Example(
        sid=0,
        edit_kind="governing",
        family="prose",
        old_ids=torch.tensor(old, device=device),
        new_ids=torch.tensor(new, device=device),
        query_ids=torch.tensor([rng.randrange(1, VOCAB) for _ in range(q)], device=device),
        qtype="obey",
        expected=None,
        span=span,
    )


# ------------------------------------------------------------------------- LoRA


def test_lora_is_identity_at_init_and_toggles(tiny):
    model, loras = tiny
    assert loras and all(isinstance(x, LoRALinear) for x in loras)
    ids = torch.arange(1, 17)[None]
    with torch.no_grad():
        with adapters(loras, False):
            off = model(input_ids=ids).logits
        with adapters(loras, True):
            on = model(input_ids=ids).logits
        assert torch.allclose(off, on)  # B initialised to zero
        for lora in loras:  # now make the adapter do something
            lora.lora_b.data.normal_(0, 0.1)
        with adapters(loras, True):
            on = model(input_ids=ids).logits
        with adapters(loras, False):
            off2 = model(input_ids=ids).logits
    assert not torch.allclose(on, off2)
    assert torch.allclose(off, off2)  # the toggle restores the base exactly
    for lora in loras:
        lora.lora_b.data.zero_()


def test_lora_state_roundtrip(tiny):
    _, loras = tiny
    for lora in loras:
        lora.lora_b.data.normal_(0, 0.1)
    state = lora_state(loras)
    saved = {k: v.clone() for k, v in state.items()}
    for lora in loras:
        lora.lora_b.data.zero_()
    load_lora_state(loras, state)
    for k, v in lora_state(loras).items():
        assert torch.equal(v, saved[k])
    for lora in loras:
        lora.lora_b.data.zero_()


def test_only_lora_params_are_trainable(tiny):
    model, loras = tiny
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    expected = {id(p) for lora in loras for p in (lora.lora_a, lora.lora_b)}
    assert trainable == expected


# ------------------------------------------------------------ differentiable stitch


def test_grad_stitch_matches_the_serving_stitch(tiny):
    """The autograd-friendly cache must place exactly what ``kvshift`` places."""
    from marathon.kvshift import inv_freq_of, stitch_segments

    model, _ = tiny
    ex = _example()
    with torch.no_grad():
        old_kv, _ = _prefill(model, ex.old_ids)
    segs = span_segments(ex.span)
    total = int(ex.new_ids.shape[0] + ex.query_ids.shape[0])
    inv = inv_freq_of(model)
    ref = stitch_segments(old_kv, segs, total, inv)
    got = grad_stitch(old_kv, segs, total, inv)
    for a, b in zip(ref.layers, got.layers, strict=True):
        assert torch.equal(a.keys, b.keys)
        assert torch.equal(a.values, b.values)


def test_scatter_layer_is_out_of_place_and_carries_gradient():
    box = [torch.tensor([1, 3])]
    keys = torch.zeros(1, 2, 5, 4)
    layer = GradShiftCache([(keys, keys.clone())]).layers[0]
    layer.box = box
    fresh = torch.ones(1, 2, 2, 4, requires_grad=True)
    out_k, _ = layer.update(fresh, fresh)
    assert torch.equal(keys, torch.zeros(1, 2, 5, 4))  # original buffer untouched
    out_k.sum().backward()
    assert fresh.grad is not None and float(fresh.grad.abs().sum()) > 0


def test_gradient_reaches_the_adapter_through_a_stitched_cache(tiny):
    model, loras = tiny
    ex = _example()
    with torch.no_grad(), adapters(loras, False):
        old_kv, _ = _prefill(model, ex.old_ids)
        forced, teacher_seq = (
            [1, 2, 3, 4],
            model(input_ids=torch.cat([ex.new_ids, ex.query_ids])[None]).logits[0, -5:],
        )
    with adapters(loras, True):
        student = stitched_logits(model, ex, [(k.detach(), v.detach()) for k, v in old_kv], forced)
    assert student.shape[0] == len(forced)
    loss = kl_to(teacher_seq, student)
    assert torch.isfinite(loss)
    loss.backward()
    # at init B == 0, so dL/dA is 0 by the chain rule and dL/dB is the live signal
    grads = [lora.lora_b.grad for lora in loras if lora.lora_b.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads) > 0
    for lora in loras:
        lora.lora_a.grad = lora.lora_b.grad = None


def test_gradient_reaches_the_adapter_without_prefill_grad(tiny):
    """The default (memory-cheap) path must still train something.

    With ``grad_prefill=False`` the stitched prefill is a constant, so q/o proj and the
    continuation tokens' k/v carry the whole signal. If that path ever stopped producing a
    gradient the trainer would silently no-op, which is much worse than being slow.
    """
    model, loras = tiny
    ex = _example()
    with torch.no_grad(), adapters(loras, False):
        old_kv, _ = _prefill(model, ex.old_ids)
        teacher = model(input_ids=torch.cat([ex.new_ids, ex.query_ids])[None]).logits[0, -4:]
    forced = [1, 2, 3, 4]
    with adapters(loras, True):
        student = stitched_logits(
            model, ex, [(k.detach(), v.detach()) for k, v in old_kv], forced, grad_prefill=False
        )
    assert student.shape[0] == len(forced)
    kl_to(teacher, student).backward()
    grads = [lora.lora_b.grad for lora in loras if lora.lora_b.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads) > 0
    for lora in loras:
        lora.lora_a.grad = lora.lora_b.grad = None


def test_clean_logits_reproduce_a_plain_forward(tiny):
    """The anchor path must be a genuine full recompute, not a second stitched run."""
    model, loras = tiny
    ex = _example()
    forced = [7, 11, 13]
    with torch.no_grad(), adapters(loras, False):
        got = clean_logits(model, ex, forced)
        ids = torch.cat([ex.new_ids, ex.query_ids, torch.tensor(forced[:-1])])
        ref = model(input_ids=ids[None]).logits[0, -len(forced) :]
    assert torch.allclose(got, ref, atol=1e-4)


def test_the_anchor_floor_is_an_exact_zero(tiny):
    """At identity the anchor must read 0 *exactly*, not "small".

    The metric it feeds — how much the adapter damaged clean-context behaviour — has to
    resolve differences below the eval's ~0.0015 greedy-vs-teacher-forced numerical floor,
    so teacher and anchor deliberately share ``_clean_sequence``. A refactor that gives
    them separate-but-equivalent paths reintroduces the floor silently; this catches it.
    """
    model, loras = tiny
    ex = _example()
    with torch.no_grad(), adapters(loras, False):
        forced, teacher_seq = teacher_reference(model, ex, gen_tokens=5)
        anchor = clean_logits(model, ex, forced)
    assert torch.equal(teacher_seq, anchor)  # identical logits, not merely close
    # KL of a distribution against itself is ~1e-8 from the float softmax, not 0.0
    assert float(kl_to(teacher_seq, anchor)) < 1e-6


def test_kl_is_zero_against_itself_and_positive_otherwise():
    a = torch.randn(5, 20)
    assert float(kl_to(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_to(a, torch.randn(5, 20))) > 0


# -------------------------------------------------------------------------- data


def test_build_examples_holds_out_by_seed_and_honours_gov_frac():
    from marathon.kvshift_eval import SNAPSHOT, load_corpus

    tok, corpus = ByteTokenizer(), load_corpus(SNAPSHOT)
    kw = dict(min_tokens=600, max_tokens=900, corpus=corpus)
    train_ex = build_examples(tok, "cpu", 8, seed=7001, gov_frac=1.0, **kw)
    held = build_examples(tok, "cpu", 8, seed=9001, gov_frac=1.0, **kw)
    assert train_ex and held
    assert all("governing" in e.edit_kind for e in train_ex)
    # different seeds must not produce the same sessions, or the eval is not held out
    assert [e.new_ids.tolist() for e in train_ex] != [e.new_ids.tolist() for e in held]
    mixed = build_examples(tok, "cpu", 8, seed=7001, gov_frac=0.0, **kw)
    assert mixed and not any("governing" in e.edit_kind for e in mixed)
    # the whole query pool must be represented, not just whatever sits first: a front slice
    # gave the 0.6B pilot 120 items that all asked `fact-at` and never asked `obey`
    wide = build_examples(tok, "cpu", 12, seed=7001, gov_frac=1.0, **kw)
    qtypes = {e.qtype for e in wide}
    assert len(qtypes) > 1, qtypes
    assert "obey" in qtypes, qtypes
    # and k>1 must not repeat one question inside a session
    pair = build_examples(tok, "cpu", 3, seed=7001, gov_frac=1.0, queries_per_item=2, **kw)
    by_sid: dict[int, list[str]] = {}
    for e in pair:
        by_sid.setdefault(e.sid, []).append(e.qtype)
    assert all(len(v) == len(set(v)) == 2 for v in by_sid.values()), by_sid
    for e in train_ex:  # every example must actually have a downstream suffix to go stale
        assert e.span.s > 0


# ------------------------------------------------------------------ train / eval


def test_one_training_step_runs_and_moves_the_adapter(tiny):
    model, loras = tiny
    examples = [_example(), _example(n=52, q=5)]
    before = lora_state(loras)
    log = train(
        model,
        loras,
        examples,
        lr=1e-2,
        gen_tokens=4,
        anchor_every=1,
        accum=1,
        log_every=0,
    )
    assert len(log) == 2
    assert all(r["stitch_kl"] >= 0 and r["clean_kl"] is not None for r in log)
    after = lora_state(loras)
    assert any(not torch.equal(before[k], after[k]) for k in before)
    for lora in loras:
        lora.lora_b.data.zero_()
        lora.lora_a.grad = lora.lora_b.grad = None


def test_split_backward_equals_the_summed_loss(tiny):
    """Backpropagating the two terms separately must give the same gradient as summing.

    Training does it separately so the stitched and anchor graphs never coexist (peak memory,
    which a first pilot run died on). That is only sound because the terms are independent —
    this pins the arithmetic.
    """
    model, loras = tiny
    ex = _example()
    params = [p for lora in loras for p in (lora.lora_a, lora.lora_b)]

    def grads(backward, weight):
        for q in params:
            q.grad = None
        parts = example_losses(
            model, loras, ex, 4, anchor=True, backward=backward, anchor_weight=weight
        )
        if backward is None:
            (parts["stitch_kl"] + weight * parts["clean_kl"]).backward()
        return [None if q.grad is None else q.grad.clone() for q in params]

    summed = grads(None, 3.0)
    split = grads(1.0, 3.0)
    assert any(g is not None for g in summed)
    for a, b in zip(summed, split, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6)
    for q in params:
        q.grad = None


def test_anchor_is_skipped_when_disabled(tiny):
    model, loras = tiny
    parts = example_losses(model, loras, _example(), gen_tokens=3, anchor=False)
    assert float(parts["clean_kl"]) == 0.0
    floats = example_losses(model, loras, _example(), gen_tokens=3, anchor=False, backward=1.0)
    assert isinstance(floats["stitch_kl"], float) and floats["clean_kl"] == 0.0
    assert len(parts["forced"]) == 3


def test_evaluate_and_report(tiny):
    model, loras = tiny
    rows = evaluate(model, loras, [_example(), _example(n=50, q=4)], tok=None, gen_tokens=3)
    assert len(rows) == 2
    for r in rows:
        assert r["governing"] is True
        # adapters are at identity here, so stitched student == stitched base exactly
        assert r["tuned_stitch_kl"] == pytest.approx(r["base_stitch_kl"], abs=1e-6)
        assert r["tuned_clean_kl"] == pytest.approx(0.0, abs=1e-5)
        assert json.dumps(r)  # rows must be JSONL-serialisable
    text = report(rows)
    assert "base_stitch_kl" in text and "governing" in text


# ------------------------------------------------------------ dependent-edit probe


def test_probe_examples_carry_the_dependent_edit_scenarios():
    """The 2026-08-18 study's scenarios must survive the trip into ``Example`` form."""
    from marathon.stitch_train import PROBE_SCENARIOS, probe_examples, probe_report

    exs = probe_examples(ByteTokenizer(), "cpu", turns=20)  # dep-contradict needs turn 13
    kinds = {e.edit_kind for e in exs}
    assert kinds == set(PROBE_SCENARIOS)
    instr = [e for e in exs if e.edit_kind == "dep-instruction"]
    assert {e.qtype for e in instr} == {"lang-pipeline", "lang-scheduler"}
    for e in exs:
        assert e.span.s > 0  # something downstream of the edit, or the scenario is void
        assert e.old_ids.tolist() != e.new_ids.tolist()
    rows = [
        {
            "edit_kind": e.edit_kind,
            "qtype": e.qtype,
            "base_stitch_kl": 0.1,
            "tuned_stitch_kl": 0.01,
            "tuned_clean_kl": 0.0,
            "ref_answer_ok": True,
            "base_answer_ok": False,
            "tuned_answer_ok": True,
        }
        for e in exs
    ]
    assert "dep-instruction" in probe_report(rows)


# --------------------------------------------------- do-no-harm term / checkpoints


def _nongov_example(**kw):
    ex = _example(**kw)
    return Example(
        sid=ex.sid,
        edit_kind="fact",
        family=ex.family,
        old_ids=ex.old_ids,
        new_ids=ex.new_ids,
        query_ids=ex.query_ids,
        qtype=ex.qtype,
        expected=ex.expected,
        span=ex.span,
    )


def test_preserve_term_is_zero_while_the_adapter_matches_the_base(tiny):
    """The hinge must not push a non-governing item that is already as good as the base.

    At identity the student *is* the base, so student-minus-base is 0 and the penalty must
    be exactly 0 -- otherwise the term would be a leash that forbids the incidental
    improvements, rather than the floor it is meant to be.
    """
    model, loras = tiny
    parts = example_losses(
        model, loras, _nongov_example(), gen_tokens=4, anchor=False, preserve_weight=1.0
    )
    assert parts["governing"] is False
    assert parts["base_stitch_kl"] is not None
    assert float(parts["stitch_kl"].detach()) == pytest.approx(parts["base_stitch_kl"], abs=1e-6)
    assert float(parts["penalty"].detach()) == pytest.approx(0.0, abs=1e-9)


def test_preserve_term_activates_once_the_adapter_regresses(tiny):
    model, loras = tiny
    for lora in loras:  # move the adapter off identity so the student differs from the base
        lora.lora_b.data.normal_(0, 0.05)
    parts = example_losses(
        model, loras, _nongov_example(), gen_tokens=4, anchor=False, preserve_weight=2.0
    )
    excess = float(parts["stitch_kl"].detach()) - parts["base_stitch_kl"]
    expected = max(0.0, excess) * 2.0
    assert float(parts["penalty"].detach()) == pytest.approx(expected, abs=1e-6)
    for lora in loras:
        lora.lora_b.data.zero_()


def test_governing_items_keep_the_plain_objective(tiny):
    """A governing item must still be asked to get *better*, not merely not-worse."""
    model, loras = tiny
    parts = example_losses(
        model, loras, _example(), gen_tokens=4, anchor=False, preserve_weight=1.0
    )
    assert parts["governing"] is True
    assert parts["base_stitch_kl"] is None  # no extra base pass is paid for
    assert float(parts["penalty"].detach()) == pytest.approx(
        float(parts["stitch_kl"].detach()), abs=1e-9
    )


def test_grad_prefill_cap_downgrades_long_items(tiny):
    """Over the cap the fresh span leaves the graph; the run must say so, not fail."""
    model, loras = tiny
    ex = _example()
    total = int(ex.new_ids.shape[0] + ex.query_ids.shape[0])
    over = example_losses(
        model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=total - 1
    )
    under = example_losses(
        model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=total
    )
    assert over["grad_prefill"] is False and under["grad_prefill"] is True
    assert over["tokens"] == total
    off = example_losses(model, loras, ex, 3, anchor=False, grad_prefill=False)
    assert off["grad_prefill"] is False
    uncapped = example_losses(
        model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=0
    )
    assert uncapped["grad_prefill"] is True  # 0 means no cap


def test_grad_prefill_reaches_the_fresh_spans_kv(tiny):
    """With the cap satisfied, k/v proj must actually receive gradient.

    This is the difference between the two halves of the method: the cheap path trains only
    how the model *reads* a stale suffix, this one also trains what it *writes*.
    """
    model, loras = tiny
    ex = _example()
    with torch.no_grad(), adapters(loras, False):
        old_kv, _ = _prefill(model, ex.old_ids)
        teacher = model(input_ids=torch.cat([ex.new_ids, ex.query_ids])[None]).logits[0, -4:]
    kvs = [m for name, m in model.named_modules() if name.endswith(("k_proj", "v_proj"))]
    with adapters(loras, True):
        out = stitched_logits(
            model, ex, [(k.detach(), v.detach()) for k, v in old_kv], [1, 2, 3, 4], True
        )
    kl_to(teacher, out).backward()
    assert max(float(m.lora_b.grad.abs().max()) for m in kvs if m.lora_b.grad is not None) > 0
    for lora in loras:
        lora.lora_a.grad = lora.lora_b.grad = None


def test_checkpoint_callback_fires_on_schedule_and_at_the_end(tiny):
    model, loras = tiny
    seen = []
    train(
        model,
        loras,
        [_example(), _nongov_example(n=52, q=5), _example(n=50, q=4)],
        lr=1e-3,
        gen_tokens=3,
        anchor_every=0,
        accum=1,
        log_every=0,
        checkpoint_every=2,
        on_checkpoint=lambda step, log: seen.append(step),
    )
    assert seen and seen[-1] == 3, seen  # final point always measured
    assert 2 in seen  # and the scheduled one fired
    for lora in loras:
        lora.lora_b.data.zero_()
        lora.lora_a.grad = lora.lora_b.grad = None


# ------------------------------------------- iteration 3: standing bucket, memory, rule


def test_standing_bucket_is_a_governing_instruction_flip_in_the_regime():
    """The `dep-instruction` distribution fix: probe-shaped sessions, sized like the rest."""
    from marathon.kvshift_eval import STANDING_KIND, build_standing_item

    tok = ByteTokenizer()

    def count(text):
        return len(tok.encode(text))

    placements = set()
    for sid in range(4):
        item = build_standing_item(
            sid, None, seed=9001, min_tokens=600, max_tokens=900, count_tokens=count
        )
        assert item.edit_kind == STANDING_KIND
        old = item.session.messages[item.msg_index]["content"]
        # a single-span edit that flips the standing instruction and nothing else
        assert old != item.new_content
        assert item.msg_index == 0
        instr_a, instr_b = item.meta["instruction"].split(" -> ")
        assert instr_a in old and instr_b not in old
        assert instr_b in item.new_content and instr_a not in item.new_content
        # the instruction governs a *later* answer, so there must be history after it
        assert len(item.session.messages) > 4
        # open-ended questions, no forced prefix -- the edit changes the answer's form
        assert item.queries and all(forced == "" for _, _, _, forced in item.queries)
        placements.add(item.meta["placement"])
        # the governing flag is what reuse_plan keys on
        flagged = [m for m in item.session.messages if m.get("governing")]
        assert flagged, item.meta
    assert placements == {"system", "early-user"}


def test_standing_items_join_the_population_and_get_their_own_bucket():
    from marathon.kvshift_eval import SNAPSHOT, STANDING_KIND, load_corpus
    from marathon.stitch_train import is_governing

    tok, corpus = ByteTokenizer(), load_corpus(SNAPSHOT)
    kw = dict(min_tokens=600, max_tokens=900, corpus=corpus, gov_frac=1.0)
    plain = build_examples(tok, "cpu", 10, seed=7001, **kw)
    # standing_frac=0 must leave the RNG stream exactly as it was, so every population
    # built before this argument existed still reproduces item for item
    assert [e.edit_kind for e in plain] == [
        e.edit_kind for e in build_examples(tok, "cpu", 10, seed=7001, standing_frac=0.0, **kw)
    ]
    mixed = build_examples(tok, "cpu", 10, seed=7001, standing_frac=1.0, **kw)
    assert all(e.edit_kind == STANDING_KIND for e in mixed)
    assert all(is_governing(e) for e in mixed)  # it *is* a governing edit

    # ... and the report keeps it beside the core bucket rather than folded into it
    def row(kind, gov, base, tuned):
        return dict(
            sid=0,
            edit_kind=kind,
            family="f",
            qtype="obey",
            governing=gov,
            base_stitch_kl=base,
            tuned_stitch_kl=tuned,
            tuned_clean_kl=0.001,
            ref_answer_ok=None,
            base_answer_ok=None,
            tuned_answer_ok=None,
        )

    text = report(
        [
            row("governing", True, 0.04, 0.02),
            row(STANDING_KIND, True, 0.06, 0.03),
            row("fact", False, 0.005, 0.006),
        ]
    )
    assert "standing-gov" in text and "governing+std" in text
    assert "+std governing/non-governing" in text
    # the core ratio must not have absorbed the new bucket: 0.04/0.005 = 8x
    assert "governing/non-governing mean KL ratio = 8.00x" in text


def test_memory_estimate_scales_with_tokens_and_the_grad_path(tiny):
    from marathon.stitch_train import kv_bytes_per_token, stitch_memory_estimate

    model, _ = tiny
    # 2 layers x 2 kv heads x 8 head_dim x 4 bytes (fp32) x 2 (K and V)
    assert kv_bytes_per_token(model) == 2 * 2 * 2 * 8 * 4
    cheap = stitch_memory_estimate(model, 8000, False)
    rich = stitch_memory_estimate(model, 8000, True)
    assert rich["cache_bytes"] > cheap["cache_bytes"]
    assert rich["cache_bytes"] == kv_bytes_per_token(model) * 8000 * rich["copies"]
    # linear in context length: this is what makes an 8k cap decidable before a run
    assert stitch_memory_estimate(model, 16000, True)["cache_bytes"] == 2 * rich["cache_bytes"]


def test_grad_prefill_falls_back_on_oom_rather_than_killing_the_run(tiny, monkeypatch):
    """A cap is a prediction; OOM is the measurement. One item must not lose the run."""
    import marathon.stitch_train as st

    model, loras = tiny
    ex = _example()
    calls = []
    real = st.stitched_logits

    def flaky(model_, ex_, old_kv, forced, grad_prefill=False):
        calls.append(grad_prefill)
        if grad_prefill:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(model_, ex_, old_kv, forced, grad_prefill)

    monkeypatch.setattr(st, "stitched_logits", flaky)
    parts = st.example_losses(
        model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=0
    )
    assert calls == [True, False]  # tried the expressive path, then fell back
    assert parts["grad_prefill"] is False and parts["grad_prefill_oom"] is True
    assert float(parts["stitch_kl"].detach()) >= 0.0
    # WSL reports GPU exhaustion as a plain RuntimeError, and iteration 3 lost a run to a
    # fallback that only caught the clean exception
    calls.clear()

    def wsl_flaky(model_, ex_, old_kv, forced, grad_prefill=False):
        calls.append(grad_prefill)
        if grad_prefill:
            raise RuntimeError("CUDA driver error: device not ready")
        return real(model_, ex_, old_kv, forced, grad_prefill)

    monkeypatch.setattr(st, "stitched_logits", wsl_flaky)
    parts = st.example_losses(
        model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=0
    )
    assert calls == [True, False] and parts["grad_prefill_oom"] is True
    # a RuntimeError that is *not* exhaustion must still propagate untouched
    monkeypatch.setattr(
        st,
        "stitched_logits",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("shapes do not match")),
    )
    with pytest.raises(RuntimeError, match="shapes do not match"):
        st.example_losses(
            model, loras, ex, 3, anchor=False, grad_prefill=True, grad_prefill_max_tokens=0
        )
    # an OOM on the *cheap* path is not recoverable and must propagate
    monkeypatch.setattr(
        st,
        "stitched_logits",
        lambda *a, **k: (_ for _ in ()).throw(torch.OutOfMemoryError("boom")),
    )
    with pytest.raises(torch.OutOfMemoryError):
        st.example_losses(model, loras, ex, 3, anchor=False, grad_prefill=False)


def test_select_checkpoint_follows_the_pre_registered_rule():
    from marathon.stitch_train import select_checkpoint

    def ck(step, p95, non_med, clean, base_med=0.0020):
        return dict(
            step=step, gov_p95=p95, non_tuned_median=non_med, non_base_median=base_med, clean=clean
        )

    history = [
        ck(50, 0.030, 0.0021, 0.0015),  # feasible
        ck(100, 0.012, 0.0022, 0.0018),  # feasible and the best tail  (0.0022 <= 0.0024)
        ck(150, 0.008, 0.0030, 0.0018),  # best tail but non-gov median blows the 1.2x bar
        ck(200, 0.009, 0.0021, 0.0031),  # best-ish tail but clean drift over 0.002
    ]
    assert select_checkpoint(history)["step"] == 100
    # the rule is allowed to select nothing, and that is a result rather than an error
    assert select_checkpoint([ck(50, 0.01, 0.9, 0.9)]) is None
    # a run with no mid-eval rows has nothing to select from
    assert select_checkpoint([{"step": 50}]) is None


# ------------------------------------ 2026-08-20: the gated statistic and its assumptions


def _row(kind, gov, base, tuned, stable=True, qtype="obey"):
    return dict(
        sid=0,
        edit_kind=kind,
        family="f",
        qtype=qtype,
        governing=gov,
        ref_stable=stable,
        base_stitch_kl=base,
        tuned_stitch_kl=tuned,
        tuned_clean_kl=0.001,
        ref_answer_ok=None,
        base_answer_ok=None,
        tuned_answer_ok=None,
    )


def test_trimmed_mean_drops_both_tails():
    from marathon.stitch_train import trimmed_mean

    # one wild outlier at each end; the 20% trim removes them
    vals = [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
    assert trimmed_mean(vals, 0.2) == pytest.approx(4.5)
    assert trimmed_mean(vals, 0.0) == pytest.approx(statistics.fmean(vals))
    # a trim that would empty the sample keeps it rather than dividing by zero
    assert trimmed_mean([1.0, 2.0], 0.5) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        trimmed_mean([])


def test_bootstrap_ci_is_seeded_brackets_the_mean_and_narrows_with_n():
    from marathon.stitch_train import bootstrap_ci

    rng = random.Random(0)
    sample = [rng.gauss(-0.01, 0.02) for _ in range(60)]
    lo, hi = bootstrap_ci(sample, resamples=2000)
    assert lo < statistics.fmean(sample) < hi
    # seeded: the same rows must give the same interval, or a reported CI is not checkable
    assert bootstrap_ci(sample, resamples=2000) == (lo, hi)
    # more data, tighter interval
    big = [rng.gauss(-0.01, 0.02) for _ in range(600)]
    lo2, hi2 = bootstrap_ci(big, resamples=2000)
    assert (hi2 - lo2) < (hi - lo)
    # a degenerate sample is a point, not a crash
    assert bootstrap_ci([0.5]) == (0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_delta_summary_pairs_per_item_and_flags_significance():
    from marathon.stitch_train import delta_summary, paired_deltas

    # every item improves by exactly 0.01 -> the interval must exclude zero
    rows = [_row("governing", True, 0.05 + i / 100, 0.04 + i / 100) for i in range(30)]
    assert paired_deltas(rows) == pytest.approx([-0.01] * 30)
    d = delta_summary(rows, resamples=2000)
    assert d["n"] == 30 and d["improved"] == 30
    assert d["mean"] == pytest.approx(-0.01) and d["significant"] is True
    assert d["ci_hi"] < 0
    # noise around zero must NOT be called significant
    rng = random.Random(7)
    noisy = [
        _row("governing", True, b := rng.uniform(0.001, 0.05), b + rng.gauss(0, 0.004))
        for _ in range(40)
    ]
    assert delta_summary(noisy, resamples=2000)["significant"] is False


def test_the_paired_delta_survives_what_the_ratio_of_means_does_not():
    """The point of the rewrite, as a test.

    Iteration 3's failure mode: a shared per-item offset (which branch of a near-tie the
    teacher took) moves both columns together. It swamps a ratio of independently measured
    means and cancels exactly in the paired delta.
    """
    from marathon.stitch_train import delta_summary

    rng = random.Random(11)
    truth = [(rng.uniform(0.002, 0.03), -0.004) for _ in range(40)]
    clean = [_row("governing", True, b, b + d) for b, d in truth]
    # now let a quarter of the items' references flip, adding a large shared offset
    shifted = []
    for i, (b, d) in enumerate(truth):
        off = 0.4 if i % 4 == 0 else 0.0
        shifted.append(_row("governing", True, b + off, b + d + off))
    a, c = delta_summary(clean, resamples=2000), delta_summary(shifted, resamples=2000)
    assert a["mean"] == pytest.approx(c["mean"], abs=1e-12)  # identical, offset cancels
    assert c["significant"] is True
    # the same perturbation moves the ratio-style bucket mean by an order of magnitude
    plain = statistics.fmean(r["base_stitch_kl"] for r in clean)
    hit = statistics.fmean(r["base_stitch_kl"] for r in shifted)
    assert hit > 5 * plain


def test_report_excludes_unstable_items_from_the_gate_and_says_so():
    from marathon.stitch_train import report, stable_rows

    rows = [_row("governing", True, 0.01, 0.008) for _ in range(6)]
    rows += [_row("governing", True, 0.40, 0.39, stable=False)]  # a flipped reference
    rows += [_row("fact", False, 0.004, 0.004) for _ in range(6)]
    assert len(stable_rows(rows)) == 12
    text = report(rows)
    assert "reference stability: 12/13 stable" in text
    assert "bucket (stable only)" in text
    assert "[descriptive only]" in text  # the ratio is demoted, not deleted
    # the gated governing row must be computed on 6 items, not 7
    lines = text.split("\n")
    gate = lines[next(i for i, ln in enumerate(lines) if "bucket (stable only)" in ln) :]
    gov_line = next(ln for ln in gate if ln.startswith("governing "))
    assert gov_line.split()[1] == "6", gov_line


def test_unprobed_rows_count_as_stable():
    """`ref_stable=None` means "not measured", which must not silently empty the gate."""
    from marathon.stitch_train import report, stable_rows

    rows = [_row("governing", True, 0.01, 0.009, stable=None) for _ in range(4)]
    rows += [_row("fact", False, 0.004, 0.004, stable=None) for _ in range(4)]
    assert len(stable_rows(rows)) == 8
    assert "reference stability" not in report(rows)


def test_chunked_prefill_matches_the_single_shot_one(tiny):
    """The perturbation must be mathematically benign: same tokens, same maths."""
    from marathon.stitch_train import _prefill, _prefill_chunked, greedy_tokens

    model, _ = tiny
    ids = torch.arange(3, 43) % VOCAB
    with torch.no_grad():
        (kv_a, log_a), (kv_b, log_b) = _prefill(model, ids), _prefill_chunked(model, ids, 3)
    assert len(kv_a) == len(kv_b)
    for (ka, va), (kb, vb) in zip(kv_a, kv_b, strict=True):
        assert ka.shape == kb.shape
        assert torch.allclose(ka, kb, atol=1e-4) and torch.allclose(va, vb, atol=1e-4)
    assert torch.allclose(log_a, log_b, atol=1e-4)
    # and in fp32 on CPU it is stable, so the probe reports stable rather than crying wolf
    assert greedy_tokens(model, ids, 4, 1) == greedy_tokens(model, ids, 4, 3)


def test_reference_stability_probe_runs_and_evaluate_records_it(tiny):
    from marathon.stitch_train import reference_is_stable, teacher_reference

    model, loras = tiny
    ex = _example()
    forced, _ = teacher_reference(model, ex, 3)
    assert reference_is_stable(model, ex, forced) is True  # fp32 CPU: nothing to flip
    # a reference that is *not* what the perturbed run produces must read unstable
    assert reference_is_stable(model, ex, [(forced[0] + 1) % VOCAB, *forced[1:]]) is False
    rows = evaluate(model, loras, [ex], None, 3, ref_stability=True)
    assert rows[0]["ref_stable"] is True
    assert evaluate(model, loras, [ex], None, 3)[0]["ref_stable"] is None  # off by default
    # rows stream out as they are produced, so a killed run keeps what it paid for
    seen = []
    evaluate(model, loras, [ex, ex], None, 3, on_row=seen.append)
    assert len(seen) == 2 and seen[0]["base_stitch_kl"] >= 0


# --------------------------------------------- the paged population (Track L's shape)


def test_paged_examples_are_multi_segment_paging_transitions():
    """The point of the population: several disjoint edits, not one contiguous rewrite."""
    from marathon.kvshift import Segment, token_span
    from marathon.kvshift_eval import SNAPSHOT, load_corpus
    from marathon.paged_eval import build_paged_examples

    tok, corpus = ByteTokenizer(), load_corpus(SNAPSHOT)
    exs = build_paged_examples(
        tok, "cpu", 6, seed=4242, turns=24, active_tokens=9000, corpus=corpus
    )
    assert exs, "no paged items built"
    for e in exs:
        assert e.edit_kind == "paged" and e.qtype == "paged-fact"
        # an explicit multi-segment plan, which is what token_span cannot express
        assert isinstance(e.span, list) and len(e.span) >= 2
        assert all(isinstance(s, Segment) for s in e.span)
        # segments are in destination order and none overlaps its neighbour
        for a, b in zip(e.span, e.span[1:], strict=False):
            assert a.dst_end <= b.dst_start
        # the reuse must reach the end, or there is no stale suffix to be wrong about
        assert e.span[-1].dst_end >= int(e.new_ids.shape[0]) - 1
        # the churn really is churn: a single-span view of the same edit would give up
        # far more of the history than the multi-segment plan does
        reused = sum(s.length for s in e.span)
        span = token_span(e.old_ids.tolist(), e.new_ids.tolist())
        assert reused > span.p + span.s, (reused, span.p + span.s)
        assert e.expected and e.meta["promoted"] >= 1
    # the stubs the cold tier writes must actually be in the view being reused
    assert all("[cold #" in tok.decode(e.old_ids.tolist()) for e in exs)


def test_paged_examples_run_through_the_stitched_forward(tiny):
    """Multi-segment plans must flow through the same graph as single-span ones."""
    from marathon.kvshift_eval import SNAPSHOT, load_corpus
    from marathon.paged_eval import build_paged_examples
    from marathon.stitch_train import example_segments

    model, loras = tiny
    tok, corpus = ByteTokenizer(), load_corpus(SNAPSHOT)
    exs = build_paged_examples(tok, "cpu", 3, seed=99, turns=24, active_tokens=9000, corpus=corpus)
    if not exs:
        pytest.skip("no paged items at this size")
    ex = exs[0]
    ex = Example(
        ex.sid,
        ex.edit_kind,
        ex.family,
        ex.old_ids % VOCAB,
        ex.new_ids % VOCAB,
        ex.query_ids % VOCAB,
        ex.qtype,
        ex.expected,
        ex.span,
        ex.meta,
    )
    assert len(example_segments(ex)) >= 2
    rows = evaluate(model, loras, [ex], None, 3)
    assert rows[0]["base_stitch_kl"] >= 0 and rows[0]["edit_kind"] == "paged"


def test_example_segments_accepts_both_plan_shapes():
    from marathon.kvshift import Segment, token_span
    from marathon.stitch_train import example_segments

    ex = _example()
    assert example_segments(ex) == span_segments(ex.span)
    segs = [Segment(0, 4, 0), Segment(8, 12, 6)]
    ex2 = Example(0, "paged", "paged", ex.old_ids, ex.new_ids, ex.query_ids, "q", None, segs)
    assert example_segments(ex2) is segs
    assert isinstance(token_span([1, 2, 3], [1, 9, 3]), object)


def test_paged_items_are_trained_as_targets_not_guarded(tiny):
    """A paged item must get the improvement objective, not the do-no-harm hinge.

    With the hinge it would contribute zero gradient whenever it is already no worse than
    the base -- i.e. the population built to be repaired would train nothing.
    """
    from marathon.stitch_train import is_governing, is_target

    model, loras = tiny
    ex = _example()
    paged = Example(
        1, "paged", "paged", ex.old_ids, ex.new_ids, ex.query_ids, "paged-fact", None, ex.span
    )
    assert not is_governing(paged) and is_target(paged)
    plain = _nongov_example()
    assert not is_target(plain)
    # the guarded item reports a base column (the hinge needs it); the target does not
    got = example_losses(model, loras, paged, 3, anchor=False, preserve_weight=2.0)
    assert got["base_stitch_kl"] is None
    assert (
        example_losses(model, loras, plain, 3, anchor=False, preserve_weight=2.0)["base_stitch_kl"]
        is not None
    )


def test_oom_during_backward_falls_back_without_double_counting_the_gradient(tiny, monkeypatch):
    """The retry must cover the backward, and must not count a half-done one twice.

    The 2026-08-20 mixed retrain died in `Tensor.backward` while an OOM handler that only
    wrapped the forward watched it happen. The fake here accumulates real gradient and
    *then* raises, which is the case that makes a naive retry double-count.
    """
    import marathon.stitch_train as st

    model, loras = tiny
    ex = _example()

    def clear():
        for lora in loras:
            lora.lora_a.grad = lora.lora_b.grad = None

    def seed():
        """A known pre-existing gradient, as an accumulation window would carry."""
        clear()
        for i, lora in enumerate(loras):
            lora.lora_b.grad = torch.full_like(lora.lora_b, 0.01 * (i + 1))

    # reference: what a clean cheap-path item adds on top of the seed
    seed()
    st.example_losses(model, loras, ex, 3, anchor=False, backward=1.0, grad_prefill=False)
    want = [lora.lora_b.grad.clone() for lora in loras]

    real_backward = torch.Tensor.backward
    fired: list[int] = []

    def flaky(self, *a, **k):
        real_backward(self, *a, **k)  # really accumulate...
        if not fired:
            fired.append(1)
            raise torch.OutOfMemoryError("CUDA out of memory")  # ...then die

    monkeypatch.setattr(torch.Tensor, "backward", flaky)
    seed()
    parts = st.example_losses(
        model,
        loras,
        ex,
        3,
        anchor=False,
        backward=1.0,
        grad_prefill=True,
        grad_prefill_max_tokens=0,
    )
    monkeypatch.undo()
    assert fired, "the fake never raised -- the test is not exercising the retry"
    assert parts["grad_prefill_oom"] is True and parts["grad_prefill"] is False
    got = [lora.lora_b.grad for lora in loras]
    for g, w in zip(got, want, strict=True):
        assert torch.allclose(g, w, atol=1e-6), "partial backward leaked into the retry"
    clear()
