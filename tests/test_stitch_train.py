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
