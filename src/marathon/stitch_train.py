"""Stitched-KV consistency fine-tuning: teach the model to read a stitched cache.

Phase 1 showed position-shifted KV reuse is near-free and near-exact *except* when the
edit lands on a **governing** span (system prompt / standing instruction). Then the
reused suffix ``S`` carries attention to the instruction that is no longer there, and KL
against full recompute jumps ~9x with wrong answers. Selective recompute does not fix it
(the whole suffix is contaminated, not a prefix of it), so the repair is either "recompute
everything after a governing edit" (correct, and gives up the win exactly where sessions
are longest) or "make the model robust to the stale attention". This module is the second.

Method — self-distillation against the model's own clean-context behaviour:

    teacher   frozen base model, LoRA disabled, full recompute of the *edited* context
    student   same weights + LoRA, forward run against the **stitched** cache
              (P verbatim, E' fresh, S re-rotated by delta -- exactly ``kvshift`` serving)
    loss      KL(teacher || student) over ``--gen-tokens`` continuation positions,
              teacher-forced on the teacher's own greedy continuation (the eval's
              ``klmean``), plus an anchor term: the same KL with the student run on a
              *clean* (unstitched) cache, which is 0 for the base model by construction
              and so directly penalises any damage to clean-context behaviour.

The teacher signal costs nothing extra to make: it is the same full-recompute reference
``kvshift_eval`` already computes, so "train" and "eval" measure the same quantity.

Gradients. RoPE re-rotation is a fixed rotation and the scatter is a copy, so stitched
attention is differentiable end to end. The reused KV itself is a constant (it came from a
previous turn; nothing can backprop into it), but the freshly computed span ``E'`` and the
query tokens *are* written into the cache inside the graph, so gradients reach both how
the model reads the stale suffix (q/o proj) and what it writes into the new span (k/v
proj). :class:`GradScatterLayer` is :class:`~marathon.kvshift.ShiftCache`'s scatter made
out-of-place for that reason.

LoRA is ~40 lines below rather than a ``peft`` dependency: it is four wrapped projections
and an enable flag, and the flag is what lets teacher and student be the same weights in
the same 8B-sized allocation.

    python -m marathon.stitch_train train --model Qwen/Qwen3-0.6B --items 600 --out lora.pt
    python -m marathon.stitch_train eval  --model Qwen/Qwen3-0.6B --lora lora.pt --items 84

CPU-testable: ``tests/test_stitch_train.py`` runs a full train step and an eval row on a
randomly initialised tiny Qwen3, so the data/loss/gradient plumbing is covered without a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from .kvshift import (
    _causal_mask,
    fresh_positions,
    inv_freq_of,
    rerotate_keys,
    span_segments,
    token_span,
)
from .kvshift_eval import EDIT_KINDS, FAMILIES, build_item, question_text, render

# --------------------------------------------------------------------------- LoRA

#: Projections we adapt. ``q_proj``/``o_proj`` change how the model *reads* the stitched
#: cache; ``k_proj``/``v_proj`` change what the fresh span writes into it. Both halves of
#: the failure mode are reachable only if all four are trained.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(x))`` with a frozen base and a runtime enable flag."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale, self.enabled = r, alpha / r, True
        dtype = torch.float32 if base.weight.dtype == torch.float32 else torch.bfloat16
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # B stays 0: identity at init
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.drop(x).to(self.lora_a.dtype)
        return out + (h @ self.lora_a.T @ self.lora_b.T).to(out.dtype) * self.scale


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.0) -> list[LoRALinear]:
    """Wrap every attention projection in ``LORA_TARGETS``; freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped: list[LoRALinear] = []
    for module in model.modules():
        for name in LORA_TARGETS:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, r, alpha, dropout).to(child.weight.device)
                setattr(module, name, lora)
                wrapped.append(lora)
    if not wrapped:  # pragma: no cover - guards a model with unexpected attention naming
        raise RuntimeError(f"no {LORA_TARGETS} projections found to adapt")
    return wrapped


@contextlib.contextmanager
def adapters(loras: list[LoRALinear], on: bool):
    """Toggle every adapter. ``on=False`` turns the student back into the teacher."""
    prev = [lora.enabled for lora in loras]
    for lora in loras:
        lora.enabled = on
    try:
        yield
    finally:
        for lora, was in zip(loras, prev, strict=True):
            lora.enabled = was


def lora_state(loras: list[LoRALinear]) -> dict[str, torch.Tensor]:
    return {
        f"{i}.{n}": p.detach().cpu().clone()
        for i, lora in enumerate(loras)
        for n, p in (("a", lora.lora_a), ("b", lora.lora_b))
    }


def load_lora_state(loras: list[LoRALinear], state: dict[str, torch.Tensor]) -> None:
    for i, lora in enumerate(loras):
        for n, p in (("a", lora.lora_a), ("b", lora.lora_b)):
            p.data.copy_(state[f"{i}.{n}"].to(p.device, p.dtype))


# ------------------------------------------------------------- differentiable stitch


class GradScatterLayer(nn.Module):
    """``ShiftCache``'s scatter layer, out-of-place so autograd can pass through it.

    The in-place ``keys[:, :, index] = ...`` in :mod:`marathon.kvshift` is fine under
    ``no_grad`` and is what the serving path wants; here the freshly written K/V must stay
    in the graph, so the write is an ``index_copy`` that returns a new tensor.
    """

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, box: list) -> None:
        super().__init__()
        self.keys, self.values, self.box = keys, values, box
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        index = self.box[0]
        if index is None:  # decode / continuation: plain append
            self.keys = torch.cat([self.keys, key_states.to(self.keys.dtype)], dim=-2)
            self.values = torch.cat([self.values, value_states.to(self.values.dtype)], dim=-2)
        else:
            self.keys = self.keys.index_copy(-2, index, key_states.to(self.keys.dtype))
            self.values = self.values.index_copy(-2, index, value_states.to(self.values.dtype))
        return self.keys, self.values

    def get_seq_length(self, *args, **kwargs) -> int:  # pragma: no cover - transformers API
        return int(self.keys.shape[-2])

    def get_mask_sizes(self, cache_position, *args, **kwargs):  # pragma: no cover
        return int(self.keys.shape[-2]), int(self.keys.shape[-2])

    def lazy_initialization(self, *args, **kwargs) -> None:  # pragma: no cover
        return None


class GradShiftCache(nn.Module):
    """The tensor-list half of :class:`~marathon.kvshift.ShiftCache`, autograd-friendly."""

    is_compileable = False

    def __init__(self, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()
        self._box: list = [None]
        self.layers = [GradScatterLayer(k, v, self._box) for k, v in kv]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i):  # pragma: no cover - transformers convenience API
        return self.layers[i].keys, self.layers[i].values

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.layers[layer_idx].keys.shape[-2])

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):  # pragma: no cover
        n = int(self.layers[layer_idx].keys.shape[-2])
        return n, n

    @property
    def index(self):
        return self._box[0]

    @index.setter
    def index(self, value) -> None:
        self._box[0] = value

    def detached(self) -> GradShiftCache:
        return GradShiftCache([(lyr.keys.detach(), lyr.values.detach()) for lyr in self.layers])

    def clone(self) -> GradShiftCache:
        return GradShiftCache([(lyr.keys.clone(), lyr.values.clone()) for lyr in self.layers])


def grad_stitch(old_kv, segments, total: int, inv_freq: torch.Tensor) -> GradShiftCache:
    """:func:`~marathon.kvshift.stitch_segments` producing a differentiable cache."""
    out = []
    for k, v in old_kv:
        nk = k.new_zeros((k.shape[0], k.shape[1], total, k.shape[3]))
        nv = v.new_zeros((v.shape[0], v.shape[1], total, v.shape[3]))
        for seg in segments:
            if seg.length <= 0:
                continue
            src, dst = slice(seg.src_start, seg.src_end), slice(seg.dst_start, seg.dst_end)
            nk[:, :, dst] = rerotate_keys(k[:, :, src], seg.delta, inv_freq)
            nv[:, :, dst] = v[:, :, src]
        out.append((nk, nv))
    return GradShiftCache(out)


def _forward_at(
    model,
    cache,
    ids: torch.Tensor,
    positions: torch.Tensor,
    total: int,
    keep: int,
    scatter: bool = True,
):
    """One masked forward through ``cache``; returns the last ``keep`` logits.

    ``scatter`` writes the fresh K/V into the pre-sized slots named by ``positions`` (the
    stitched prefill); otherwise they are appended, which is what a continuation past the
    end of the buffer needs.
    """
    cache.index = positions if scatter else None
    out = model(
        input_ids=ids[None],
        attention_mask=_causal_mask(positions, total, model.dtype),
        position_ids=positions[None],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=keep,
    )
    cache.index = None
    return out.logits[0]


def _continue_forced(model, cache, forced: list[int], seen: int, device):
    """Teacher-forced continuation logits appended to ``cache`` (which grows by ``len``)."""
    n = len(forced)
    positions = torch.arange(seen, seen + n, device=device)
    ids = torch.tensor(forced, device=device)
    return _forward_at(model, cache, ids, positions, seen + n, n, scatter=False)


def sequence_logits(
    model, cache, ids, positions, total, forced: list[int], scatter: bool = True
) -> torch.Tensor:
    """Logits for ``[first continuation step, *teacher-forced steps]`` — the eval's shape.

    ``cache`` is consumed (grown by the continuation), so pass a clone if you need it.
    """
    first = _forward_at(model, cache, ids, positions, total, 1, scatter=scatter)
    if len(forced) <= 1:
        return first
    rest = _continue_forced(model, cache, forced[:-1], total, ids.device)
    return torch.cat([first, rest], dim=0)


def kl_to(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Mean over positions of ``KL(teacher || student)``. Matches the eval's ``klmean``."""
    n = min(teacher_logits.shape[0], student_logits.shape[0])
    p = torch.softmax(teacher_logits[:n].float(), dim=-1)
    logq = torch.log_softmax(student_logits[:n].float(), dim=-1)
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(-1).mean()


# ----------------------------------------------------------------------- data


@dataclass
class Example:
    """One staged edit, tokenised, with the query the loss is measured on."""

    sid: int
    edit_kind: str
    family: str
    old_ids: torch.Tensor
    new_ids: torch.Tensor
    query_ids: torch.Tensor
    qtype: str
    expected: list[str] | None
    span: object


def build_examples(
    tok,
    device,
    n_items: int,
    seed: int,
    gov_frac: float = 0.5,
    min_tokens: int = 4000,
    max_tokens: int = 8000,
    queries_per_item: int = 1,
    corpus=None,
) -> list[Example]:
    """Stage ``n_items`` (session, edit) pairs from :mod:`marathon.kvshift_eval` builders.

    ``gov_frac`` of them get a governing edit (the failure class under repair); the rest are
    drawn from the other kinds as regularisation, so the adapter cannot buy governing
    robustness by wrecking the cases that already work. ``seed`` selects the population —
    train and eval must use different seeds, and the eval seed is never trained on.

    ``queries_per_item`` questions are taken per session, rotating the entry point into
    the pool by session id so the population covers every query type even at ``k=1``.
    """
    from .kvshift_eval import load_corpus

    corpus = corpus if corpus is not None else load_corpus()
    other = [k for k in EDIT_KINDS if "governing" not in k]
    rng = random.Random(seed)
    out: list[Example] = []
    for sid in range(n_items):
        kind = (
            ("governing" if sid % 2 else "mid-governing")
            if rng.random() < gov_frac
            else other[sid % len(other)]
        )
        item = build_item(
            sid,
            kind,
            FAMILIES[sid % len(FAMILIES)],
            corpus,
            seed,
            min_tokens,
            max_tokens,
            lambda s: len(tok.encode(s, add_special_tokens=False)),
        )
        old_text = render(tok, item.session.messages)
        item.session.edit(item.msg_index, item.new_content)
        new_text = render(tok, item.session.messages)

        def enc(text):
            return torch.tensor(tok.encode(text, add_special_tokens=False), device=device)

        old_ids, new_ids = enc(old_text), enc(new_text)
        span = token_span(old_ids.tolist(), new_ids.tolist())
        if span.s == 0:  # pragma: no cover - nothing downstream, nothing to be stale
            continue
        # Rotate the entry point into the query pool by session id rather than slicing from
        # the front. ``kvshift_eval._queries`` always puts ``fact-at`` first and, for
        # governing edits, ``obey`` last — so a front slice asks the *same* question for
        # every item, which is exactly what made the 0.6B pilot's 120 eval items all
        # ``fact-at`` and left ``obey`` (the query a governing edit most directly targets)
        # untested. Rotation is deterministic and spreads every position evenly over the
        # population, so a k=1 run still covers the whole pool across items.
        pool = item.queries
        picks = [pool[(sid + j) % len(pool)] for j in range(min(queries_per_item, len(pool)))]
        for qtype, expected, question, forced_prefix in picks:
            out.append(
                Example(
                    sid=sid,
                    edit_kind=kind,
                    family=item.family,
                    old_ids=old_ids,
                    new_ids=new_ids,
                    query_ids=enc(
                        question_text(tok, item.session.messages, question, forced_prefix)
                    ),
                    qtype=qtype,
                    expected=expected,
                    span=span,
                )
            )
    return out


# ------------------------------------------------------------------- one example


def _prefill(model, ids: torch.Tensor):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return [(lyr.keys, lyr.values) for lyr in cache.layers], out.logits[0, -1]


def _clean_sequence(model, ids: torch.Tensor, forced: list[int]) -> torch.Tensor:
    """Logit sequence over ``forced`` from a plain full recompute of ``ids``.

    Used for *both* the teacher and the student's clean-context anchor, deliberately: the
    two then agree bit for bit when the adapter is at identity, so the anchor's floor is a
    true zero rather than the ~0.0015 the eval's `prefix-equiv` row sits at (that floor is
    the price of comparing a step-by-step greedy decode against a batched teacher-forced
    one, and it is above the damage this phase needs to resolve). The prefill runs under
    ``no_grad`` — the anchor's gradient comes from the continuation positions, which is
    also what keeps an 8B anchor affordable.
    """
    with torch.no_grad():
        kv, first = _prefill(model, ids)
    cache = GradShiftCache([(k.detach(), v.detach()) for k, v in kv])
    if len(forced) <= 1:
        return first[None]
    rest = _continue_forced(model, cache, forced[:-1], int(ids.shape[0]), ids.device)
    return torch.cat([first[None], rest], dim=0)


@torch.no_grad()
def teacher_reference(model, ex: Example, gen_tokens: int) -> tuple[list[int], torch.Tensor]:
    """Greedy continuation and its logit sequence, from a full recompute of the new context.

    The greedy tokens are ``kvshift_eval``'s ``full-recompute`` answer and the sequence is
    the same teacher-forced comparison the eval's ``klmean`` is built on, so a KL that goes
    down in training is the KL the eval reports.
    """
    ids = torch.cat([ex.new_ids, ex.query_ids])
    kv, logits = _prefill(model, ids)
    cache = GradShiftCache([(k, v) for k, v in kv])
    toks = []
    for i in range(gen_tokens):
        nxt = int(logits.argmax())
        toks.append(nxt)
        if i == gen_tokens - 1:
            break
        seen = cache.get_seq_length()
        logits = _forward_at(
            model,
            cache,
            torch.tensor([nxt], device=ids.device),
            torch.tensor([seen], device=ids.device),
            seen + 1,
            1,
            scatter=False,
        )[-1]
    del cache, kv
    return toks, _clean_sequence(model, ids, toks)


def stitched_logits(
    model, ex: Example, old_kv, forced: list[int], grad_prefill: bool = False
) -> torch.Tensor:
    """Student logits over the continuation, run against the stitched cache.

    ``grad_prefill`` decides how far back gradients reach, and it is a memory decision as
    much as a modelling one:

    * ``False`` (default) — the stitched prefill of ``E'`` and the query runs under
      ``no_grad`` and the cache is detached, so gradients flow only through the
      continuation forward. This is the arrangement the phase was specified with: the
      reused KV is a constant, and what is being trained is how the model *reads* a stale
      suffix (q/o proj, plus k/v of the continuation tokens).
    * ``True`` — the fresh span's K/V stay in the graph, so ``k_proj``/``v_proj`` also learn
      what to *write* so the stale suffix matters less. Strictly more expressive, and it
      costs a retained full-length K and V per layer (~2.4 GB at 8B/8k on top of the
      attention activations). At 8B/4-8k that is what exhausted the GPU 81 items into the
      first training run — WSL reports the exhaustion as ``dxgk ... Ioctl failed: -12``
      surfacing in torch as "CUDA driver error: device not ready", not as a clean OOM.
    """
    all_ids = torch.cat([ex.new_ids, ex.query_ids])
    total = int(all_ids.shape[0])
    segments = span_segments(ex.span)
    cache = grad_stitch(old_kv, segments, total, inv_freq_of(model))
    positions = fresh_positions(segments, total, all_ids.device)
    if grad_prefill:
        return sequence_logits(model, cache, all_ids[positions], positions, total, forced)
    with torch.no_grad():
        first = _forward_at(model, cache, all_ids[positions], positions, total, 1)
    if len(forced) <= 1:
        return first
    rest = _continue_forced(model, cache.detached(), forced[:-1], total, all_ids.device)
    return torch.cat([first, rest], dim=0)


def clean_logits(model, ex: Example, forced: list[int]) -> torch.Tensor:
    """Student logits over the same continuation from a *clean* full-recompute context.

    The anchor term. With the adapter at identity this is the teacher, bit for bit — same
    function, same weights — so its KL is exactly 0 and any nonzero value is adapter-induced
    drift on an ordinary context, in the same units as the stitched KL it is traded against.
    """
    return _clean_sequence(model, torch.cat([ex.new_ids, ex.query_ids]), forced)


def is_governing(ex: Example) -> bool:
    """Whether this example edits a span that governs later generation."""
    return "governing" in ex.edit_kind


def example_losses(
    model,
    loras: list[LoRALinear],
    ex: Example,
    gen_tokens: int,
    anchor: bool,
    backward: float | None = None,
    anchor_weight: float = 1.0,
    grad_prefill: bool = False,
    preserve_weight: float = 0.0,
    preserve_slack: float = 0.0,
    grad_prefill_max_tokens: int = 6000,
) -> dict:
    """(stitched KL, clean-anchor KL) for one example. Teacher is the same weights, off.

    ``preserve_weight`` turns on the **do-no-harm term** for non-governing items, which is
    the direct answer to the 8B run's one real regression: governing KL fell 43% while
    non-governing rose 37%, even though non-governing items were half the training mix.
    They were in the mix, but they could not defend themselves — a non-governing item starts
    at KL ~0.005 against a governing item's ~0.037, so plain ``stitch_kl`` gives the cases
    that are already fine almost no gradient, and the optimiser is free to spend them.

    The fix is to change what is asked of them. A governing item is asked to get *better*
    (minimise ``stitch_kl``); a non-governing item is asked only not to get *worse* than the
    frozen base model already is on that same item:

        loss = relu(stitch_kl_student - stitch_kl_base - slack)

    which is exactly zero while the adapter is at least as good as the base and grows only
    when it regresses. That costs one extra base-model stitched forward per non-governing
    item (adapters off, no grad), and it is a hinge rather than a KL-to-base term on purpose:
    matching the base *exactly* on non-governing items would also forbid the incidental
    improvements, and the goal is a floor, not a leash.

    ``backward`` is the gradient scale (``1/accum``); when given, each term is backpropagated
    and freed *as soon as it is computed* rather than summed first. That is not a style
    choice — one 5.5k-token stitched cache is ~0.6 GB, the out-of-place scatter retains
    another ~0.6 GB in the graph, and holding the stitched and anchor graphs simultaneously
    roughly doubles the peak. Summing them first is what a first pilot run did, and it
    degraded from 1.9 s to 19 s per item before dying on a CUDA error; the two terms are
    independent, so backpropagating them separately is arithmetically identical and halves
    the footprint. With ``backward=None`` the tensors are returned instead, for tests.
    """
    governing = is_governing(ex)
    guard = preserve_weight > 0 and not governing
    base_kl = None
    with adapters(loras, False), torch.no_grad():
        forced, teacher_seq = teacher_reference(model, ex, gen_tokens)
        if guard:
            base_old_kv, _ = _prefill(model, ex.old_ids)
            base_kl = float(
                kl_to(
                    teacher_seq,
                    stitched_logits(
                        model, ex, [(k, v) for k, v in base_old_kv], forced, grad_prefill=False
                    ),
                )
            )
            del base_old_kv
    with adapters(loras, True):
        with torch.no_grad():
            old_kv, _ = _prefill(model, ex.old_ids)
            old_kv = [(k.detach(), v.detach()) for k, v in old_kv]
        # `grad_prefill` is capped by context length rather than checkpointed. Gradient
        # checkpointing would re-run each decoder layer during backward, and those re-runs
        # call `Cache.update` again -- re-entering the scatter that this module's cache
        # performs -- so the safe, verifiable bound is "only keep the fresh span in the graph
        # while the context is small enough to afford it". Items over the cap silently fall
        # back to the cheap path and are counted, so a run always reports how much of the
        # expressive half of the method it actually got.
        total_tokens = int(ex.new_ids.shape[0] + ex.query_ids.shape[0])
        used_grad_prefill = grad_prefill and (
            grad_prefill_max_tokens <= 0 or total_tokens <= grad_prefill_max_tokens
        )
        stitch_kl = kl_to(
            teacher_seq, stitched_logits(model, ex, old_kv, forced, used_grad_prefill)
        )
        # governing: get better. non-governing: just do not get worse than the base already is.
        term = (
            torch.clamp(stitch_kl - (base_kl + preserve_slack), min=0.0) * preserve_weight
            if guard
            else stitch_kl
        )
        if backward is not None:
            if term.requires_grad:
                (term * backward).backward()
            term = float(term.detach())
            stitch_kl = float(stitch_kl.detach())
        del old_kv

        if anchor:
            clean_kl = kl_to(teacher_seq, clean_logits(model, ex, forced))
            if backward is not None:
                (clean_kl * backward * anchor_weight).backward()
                clean_kl = float(clean_kl.detach())
        else:
            clean_kl = 0.0 if backward is not None else torch.zeros(())
    return {
        "stitch_kl": stitch_kl,
        "clean_kl": clean_kl,
        "base_stitch_kl": base_kl,
        "penalty": term,
        "governing": governing,
        "grad_prefill": used_grad_prefill,
        "tokens": total_tokens,
        "forced": forced,
    }


# --------------------------------------------------------------------------- train


def train(
    model,
    loras: list[LoRALinear],
    examples: list[Example],
    *,
    lr: float = 1e-4,
    epochs: int = 1,
    gen_tokens: int = 32,
    anchor_weight: float = 1.0,
    anchor_every: int = 2,
    accum: int = 4,
    grad_prefill: bool = False,
    preserve_weight: float = 0.0,
    preserve_slack: float = 0.0,
    grad_prefill_max_tokens: int = 6000,
    clip: float = 1.0,
    log_every: int = 20,
    checkpoint_every: int = 0,
    on_checkpoint=None,
    on_step=None,
) -> list[dict]:
    """One pass (or ``epochs``) of stitched-KV consistency fine-tuning. Returns the log."""
    params = [p for lora in loras for p in (lora.lora_a, lora.lora_b)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    log: list[dict] = []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(1234 + epoch).shuffle(order)
        for n, i in enumerate(order, 1):
            ex = examples[i]
            anchor = anchor_every > 0 and step % anchor_every == 0
            # the anchor's weight rides on its own backward; the two terms never coexist
            parts = example_losses(
                model,
                loras,
                ex,
                gen_tokens,
                anchor,
                1.0 / accum,
                anchor_weight,
                grad_prefill,
                preserve_weight,
                preserve_slack,
                grad_prefill_max_tokens,
            )
            if n % accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            row = {
                "epoch": epoch,
                "step": step,
                "edit_kind": ex.edit_kind,
                "stitch_kl": parts["stitch_kl"],
                "clean_kl": parts["clean_kl"] if anchor else None,
                "base_stitch_kl": parts["base_stitch_kl"],
                "penalty": parts["penalty"],
                "governing": parts["governing"],
                "grad_prefill": parts["grad_prefill"],
                "tokens": parts["tokens"],
                "elapsed_s": time.perf_counter() - t0,
            }
            log.append(row)
            if on_step:
                on_step(row)
            if log_every and step % log_every == 0:
                anchor_str = "-" if row["clean_kl"] is None else f"{row['clean_kl']:.4f}"
                print(
                    f"[{epoch}:{n}/{len(order)}] {ex.edit_kind:<14}"
                    f"stitch_kl={row['stitch_kl']:.4f} clean_kl={anchor_str} "
                    f"({row['elapsed_s']:.0f}s)",
                    flush=True,
                )
            step += 1
            # not every step: `empty_cache` returns every cached block to the driver, so the
            # next step re-`cudaMalloc`s the whole ~2 GB of caches. Periodic is enough to keep
            # the varying 3-6k session sizes from fragmenting the allocator.
            if ex.old_ids.device.type == "cuda" and step % 25 == 0:
                torch.cuda.empty_cache()
            # Mid-training checkpoints exist to settle a specific ambiguity from the 8B run:
            # the split-half training medians moved the *wrong* way while held-out KL improved
            # 43%, and with no intermediate evaluation there was no way to tell a hard second
            # half from an improvement that was not really coming from training.
            if checkpoint_every and on_checkpoint and step % checkpoint_every == 0:
                on_checkpoint(step, log)
    if checkpoint_every and on_checkpoint:
        on_checkpoint(step, log)  # final point, so the curve always ends on a measurement
    return log


# ---------------------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(
    model,
    loras: list[LoRALinear],
    examples: list[Example],
    tok=None,
    gen_tokens: int = 32,
    base_only: bool = False,
) -> list[dict]:
    """Per-example rows: reuse-all KL and clean-context KL, base vs adapted, plus accuracy.

    Four numbers per example, all against the *base* model's full recompute of the edited
    context — the same reference ``kvshift_eval`` uses:

        base_stitch_kl    the Phase 1 failure being repaired
        tuned_stitch_kl   the same run with the adapter on
        tuned_clean_kl    damage: adapter on, clean cache (base's is 0 by construction)
        *_answer_ok       planted-fact answer contains the expected code (when there is one)
    """
    rows = []
    for ex in examples:
        with adapters(loras, False):
            forced, teacher_seq = teacher_reference(model, ex, gen_tokens)
            old_kv, _ = _prefill(model, ex.old_ids)
            old_kv = [(k.detach(), v.detach()) for k, v in old_kv]
            base_seq = stitched_logits(model, ex, old_kv, forced)
        if base_only:
            # the adapter is at identity, so the tuned columns would only re-measure the base
            # ones at double the cost. Used for the "does the failure class even reproduce
            # here?" run that must precede spending an epoch on training.
            tuned_seq, tuned_clean = base_seq, teacher_seq
        else:
            with adapters(loras, True):
                tuned_old_kv, _ = _prefill(model, ex.old_ids)
                tuned_old_kv = [(k.detach(), v.detach()) for k, v in tuned_old_kv]
                tuned_seq = stitched_logits(model, ex, tuned_old_kv, forced)
                tuned_clean = clean_logits(model, ex, forced)

        def ok(seq, expected=ex.expected):
            if expected is None or tok is None:
                return None
            text = tok.decode(seq.argmax(-1).tolist())
            return any(e in text for e in expected)

        rows.append(
            {
                "sid": ex.sid,
                "edit_kind": ex.edit_kind,
                "family": ex.family,
                "qtype": ex.qtype,
                "governing": "governing" in ex.edit_kind,
                "base_stitch_kl": float(kl_to(teacher_seq, base_seq)),
                "tuned_stitch_kl": float(kl_to(teacher_seq, tuned_seq)),
                "tuned_clean_kl": float(kl_to(teacher_seq, tuned_clean)),
                "ref_answer_ok": ok(teacher_seq),
                "base_answer_ok": ok(base_seq),
                "tuned_answer_ok": ok(tuned_seq),
            }
        )
        if ex.old_ids.device.type == "cuda" and len(rows) % 25 == 0:
            torch.cuda.empty_cache()
    return rows


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p95": sorted(vals)[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))],
        "max": max(vals),
        "over_005": sum(v > 0.05 for v in vals),
    }


def report(rows: list[dict]) -> str:
    """The before/after table. Governing vs non-governing is the whole question."""
    buckets = [
        ("governing", [r for r in rows if r["governing"]]),
        ("non-governing", [r for r in rows if not r["governing"]]),
        ("ALL", rows),
    ]
    out = [
        f"{'bucket':<16}{'metric':<18}{'n':>4}{'mean':>10}{'median':>10}"
        f"{'p95':>10}{'max':>10}{'>.05':>6}"
    ]
    for name, rs in buckets:
        if not rs:
            continue
        for metric in ("base_stitch_kl", "tuned_stitch_kl", "tuned_clean_kl"):
            a = _agg(rs, metric)
            out.append(
                f"{name:<16}{metric:<18}{a['n']:>4}{a['mean']:>10.4f}{a['median']:>10.4f}"
                f"{a['p95']:>10.4f}{a['max']:>10.4f}{a['over_005']:>6}"
            )
    for name, rs in buckets:
        graded = [r for r in rs if r["ref_answer_ok"] is not None]
        if graded:
            out.append(
                f"{name:<16}planted-fact ok   n={len(graded):<4} "
                f"ref={sum(bool(r['ref_answer_ok']) for r in graded)} "
                f"base={sum(bool(r['base_answer_ok']) for r in graded)} "
                f"tuned={sum(bool(r['tuned_answer_ok']) for r in graded)}"
            )
    qtypes = sorted({r["qtype"] for r in rows})
    if len(qtypes) > 1:
        out.append("")
        for qt in qtypes:
            rs = [r for r in rows if r["qtype"] == qt]
            a, b = _agg(rs, "base_stitch_kl"), _agg(rs, "tuned_stitch_kl")
            out.append(
                f"{qt:<16}{'base->tuned mean':<18}{a['n']:>4}{a['mean']:>10.4f}"
                f"{b['mean']:>10.4f}   (median {a['median']:.4f} -> {b['median']:.4f})"
            )
    gov = [r for r in rows if r["governing"]]
    non = [r for r in rows if not r["governing"]]
    if gov and non:
        # both, because they say different things: the 144-session eval's headline 9x is a
        # ratio of means (driven by the heavy tail governing owns), while the median ratio
        # is ~3-4.5x and describes the typical item. A fix must move both.
        for tag in ("base", "tuned"):
            for stat, fn in (("mean", statistics.fmean), ("median", statistics.median)):
                g = fn([r[f"{tag}_stitch_kl"] for r in gov])
                n = fn([r[f"{tag}_stitch_kl"] for r in non])
                out.append(
                    f"{tag:<16}governing/non-governing {stat} KL ratio "
                    f"= {g / max(n, 1e-9):.2f}x  ({g:.4f} / {n:.4f})"
                )
    return "\n".join(out)


# --------------------------------------------------------------- dependent-edit probe

#: The hand-built scenarios from the dependent-edit study. ``dep-instruction`` is the one
#: that broke (first-token KL 0.3492 at reuse-all); the other two are controls that did
#: *not* break, and must stay unbroken.
PROBE_SCENARIOS = ("dep-instruction", "dep-anaphora", "dep-contradict")


def probe_examples(tok, device, turns: int = 20, names=PROBE_SCENARIOS) -> list[Example]:
    """``kvshift_probe``'s dependent-edit scenarios, as :class:`Example` objects.

    Same sessions, same questions, same chat-template rendering as the 2026-08-18 study,
    so the numbers are comparable to the entry in findings.md.
    """
    from . import kvshift_probe as probe

    builders = {
        "dep-instruction": probe.build_dep_instruction,
        "dep-anaphora": probe.build_dep_anaphora,
        "dep-contradict": probe.build_dep_contradict,
    }
    out: list[Example] = []
    for sid, name in enumerate(names):
        session, mutate, questions = builders[name](turns)
        old_text = probe.render(session, tok)
        mutate(session)
        new_text = probe.render(session, tok)

        def enc(text):
            return torch.tensor(tok.encode(text, add_special_tokens=False), device=device)

        old_ids, new_ids = enc(old_text), enc(new_text)
        span = token_span(old_ids.tolist(), new_ids.tolist())
        for qname, expected, question, forced_prefix, _n in questions:
            out.append(
                Example(
                    sid=sid,
                    edit_kind=name,
                    family="probe",
                    old_ids=old_ids,
                    new_ids=new_ids,
                    query_ids=enc(probe.question_text(tok, question, forced_prefix)),
                    qtype=qname,
                    expected=expected,
                    span=span,
                )
            )
    return out


def probe_report(rows: list[dict]) -> str:
    """Per-scenario, per-question table — these are single runs, not a distribution."""
    out = [
        f"{'scenario':<18}{'question':<16}{'base_kl':>10}{'tuned_kl':>10}"
        f"{'clean_kl':>10}{'ref_ok':>8}{'base_ok':>8}{'tuned_ok':>9}"
    ]
    for r in rows:
        out.append(
            f"{r['edit_kind']:<18}{r['qtype']:<16}{r['base_stitch_kl']:>10.4f}"
            f"{r['tuned_stitch_kl']:>10.4f}{r['tuned_clean_kl']:>10.4f}"
            f"{str(r['ref_answer_ok']):>8}{str(r['base_answer_ok']):>8}"
            f"{str(r['tuned_answer_ok']):>9}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------- cli


def _load(model_name: str, device: str, attn: str, r: int, alpha: int, dropout: float):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation=attn,
        )
        .to(device)
        .eval()
    )
    return tok, model, apply_lora(model, r, alpha, dropout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marathon.stitch_train", description=__doc__)
    ap.add_argument("cmd", choices=["train", "eval", "probe"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    # sdpa, like kvshift_eval: eager materialises the full [heads, q, kv] fp32 score matrix,
    # which at 8B and 8k tokens is ~8 GB for a single layer and dies before the first item.
    # Both paths honour the explicit additive mask the stitched forward passes.
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--items", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--gov-frac", type=float, default=0.5)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--queries-per-item", type=int, default=1)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--anchor-every", type=int, default=2)
    ap.add_argument("--lora", default=None, help="checkpoint to load (eval) or resume from")
    ap.add_argument("--out", default=None, help="where to write the adapter / rows")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--probe-turns", type=int, default=20)
    ap.add_argument("--base-only", action="store_true", help="skip the tuned columns")
    ap.add_argument(
        "--grad-prefill",
        action="store_true",
        help="backprop into the fresh span's K/V too (more expressive, far more memory)",
    )
    ap.add_argument(
        "--grad-prefill-max-tokens",
        type=int,
        default=6000,
        help="items longer than this fall back to the cheap path (0 = no cap)",
    )
    ap.add_argument(
        "--preserve-weight",
        type=float,
        default=0.0,
        help="do-no-harm weight on non-governing items: relu(student - base) hinge",
    )
    ap.add_argument("--preserve-slack", type=float, default=0.0)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save the adapter and run a held-out mid-training eval every N items",
    )
    ap.add_argument(
        "--mid-eval-items", type=int, default=24, help="sessions in each mid-training eval"
    )
    args = ap.parse_args(argv)

    tok, model, loras = _load(
        args.model, args.device, args.attn, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    if args.lora:
        load_lora_state(loras, torch.load(args.lora, map_location="cpu", weights_only=True))
    dev = next(model.parameters()).device

    if args.cmd == "probe":
        probes = probe_examples(tok, dev, args.probe_turns)
        rows = evaluate(model, loras, probes, tok, args.gen_tokens)
        print("\n" + probe_report(rows))
        if args.jsonl:
            with open(args.jsonl, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        return 0

    examples = build_examples(
        tok,
        dev,
        args.items,
        args.seed,
        args.gov_frac,
        args.min_tokens,
        args.max_tokens,
        args.queries_per_item,
    )
    print(
        f"{args.cmd}: {len(examples)} examples from {args.items} sessions "
        f"(seed {args.seed}, gov_frac {args.gov_frac}), model {args.model}",
        flush=True,
    )

    if args.cmd == "train":
        mid: list[Example] = []
        if args.checkpoint_every and args.mid_eval_items:
            # a *held-out* slice: same seed as the final eval, so the mid-training curve and
            # the final number are measured on the same population
            mid = build_examples(
                tok,
                dev,
                args.mid_eval_items,
                args.seed + 2000,
                args.gov_frac,
                args.min_tokens,
                args.max_tokens,
                args.queries_per_item,
            )
            print(f"mid-training eval slice: {len(mid)} held-out examples", flush=True)

        history: list[dict] = []

        def on_checkpoint(step, _log):
            path = f"{args.out}.step{step}" if args.out else None
            if path:
                torch.save(lora_state(loras), path)
            row = {"step": step}
            if mid:
                rows = evaluate(model, loras, mid, tok, args.gen_tokens)
                g = [r for r in rows if r["governing"]]
                n = [r for r in rows if not r["governing"]]
                for tag, rs in (("gov", g), ("non", n)):
                    if rs:
                        row[f"{tag}_base"] = statistics.fmean([r["base_stitch_kl"] for r in rs])
                        row[f"{tag}_tuned"] = statistics.fmean([r["tuned_stitch_kl"] for r in rs])
                row["clean"] = statistics.fmean([r["tuned_clean_kl"] for r in rows])
                print(
                    f"[checkpoint {step}] gov {row.get('gov_base', float('nan')):.4f}->"
                    f"{row.get('gov_tuned', float('nan')):.4f}  non "
                    f"{row.get('non_base', float('nan')):.4f}->"
                    f"{row.get('non_tuned', float('nan')):.4f}  clean {row['clean']:.4f}",
                    flush=True,
                )
            history.append(row)

        log = train(
            model,
            loras,
            examples,
            lr=args.lr,
            epochs=args.epochs,
            gen_tokens=args.gen_tokens,
            anchor_weight=args.anchor_weight,
            grad_prefill=args.grad_prefill,
            grad_prefill_max_tokens=args.grad_prefill_max_tokens,
            preserve_weight=args.preserve_weight,
            preserve_slack=args.preserve_slack,
            checkpoint_every=args.checkpoint_every,
            on_checkpoint=on_checkpoint if args.checkpoint_every else None,
            anchor_every=args.anchor_every,
            accum=args.accum,
        )
        if args.out:
            torch.save(lora_state(loras), args.out)
            print(f"wrote adapter -> {args.out}")
        if args.jsonl:
            with open(args.jsonl, "w", encoding="utf-8") as f:
                for row in log:
                    f.write(json.dumps(row) + "\n")
        if history:
            with open(f"{args.jsonl}.checkpoints", "w", encoding="utf-8") as f:
                for row in history:
                    f.write(json.dumps(row) + "\n")
        downgraded = sum(1 for r in log if args.grad_prefill and not r["grad_prefill"])
        if args.grad_prefill:
            print(
                f"grad-prefill: {len(log) - downgraded}/{len(log)} items kept the fresh span "
                f"in the graph (cap {args.grad_prefill_max_tokens} tokens)"
            )
        gov = [r["stitch_kl"] for r in log if "governing" in r["edit_kind"]]
        if gov:
            half = len(gov) // 2
            print(
                f"governing stitch_kl: first half median {statistics.median(gov[:half]):.4f} "
                f"-> second half median {statistics.median(gov[half:]):.4f}"
            )
        return 0

    rows = evaluate(model, loras, examples, tok, args.gen_tokens)
    print("\n" + report(rows))
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
