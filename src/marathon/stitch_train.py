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
from dataclasses import dataclass, field

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
from .kvshift_eval import (
    EDIT_KINDS,
    FAMILIES,
    STANDING_KIND,
    build_item,
    question_text,
    render,
)

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
    #: a :class:`~marathon.kvshift.Span` for a single-edit item, or an explicit list of
    #: :class:`~marathon.kvshift.Segment` for a multi-segment one (the paged population's
    #: demote-plus-promote turns are several disjoint edits, not one)
    span: object
    meta: dict = field(default_factory=dict)


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
    standing_frac: float = 0.0,
) -> list[Example]:
    """Stage ``n_items`` (session, edit) pairs from :mod:`marathon.kvshift_eval` builders.

    ``gov_frac`` of them get a governing edit (the failure class under repair); the rest are
    drawn from the other kinds as regularisation, so the adapter cannot buy governing
    robustness by wrecking the cases that already work. ``seed`` selects the population —
    train and eval must use different seeds, and the eval seed is never trained on.

    ``queries_per_item`` questions are taken per session, rotating the entry point into
    the pool by session id so the population covers every query type even at ``k=1``.

    ``standing_frac`` is the share of the *governing* half drawn as ``standing-governing``
    (:func:`~marathon.kvshift_eval.build_standing_item`) instead of ``governing`` /
    ``mid-governing``. It exists because iteration 2's adapter got *worse* on the
    `dep-instruction` probe: the probe is a homogeneous log with an early standing
    instruction and open-ended questions, which the training population never contained, so
    the probe was measuring off-distribution generalisation rather than the thing trained.
    At ``0.0`` the RNG stream is untouched (the second draw short-circuits), so populations
    built before this argument existed reproduce exactly.
    """
    from .kvshift_eval import load_corpus

    corpus = corpus if corpus is not None else load_corpus()
    other = [k for k in EDIT_KINDS if "governing" not in k]
    rng = random.Random(seed)
    out: list[Example] = []
    for sid in range(n_items):
        if rng.random() < gov_frac:
            kind = (
                STANDING_KIND
                if standing_frac > 0 and rng.random() < standing_frac
                else ("governing" if sid % 2 else "mid-governing")
            )
        else:
            kind = other[sid % len(other)]
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


def _prefill_chunked(model, ids: torch.Tensor, chunks: int = 2):
    """:func:`_prefill` with the prompt fed in ``chunks`` pieces instead of one.

    Mathematically identical — same tokens, same causal mask, same weights — but the matmul
    shapes differ, so cuBLAS may pick a different kernel and sum in a different order. In
    bf16 that is enough to move a logit by a ULP, which is enough to flip an argmax that was
    a near-tie. That is the whole point: it is a *benign* perturbation, and any answer that
    changes under it was never determined by the model in the first place.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    n = int(ids.shape[0])
    size = max(1, (n + chunks - 1) // chunks)
    logits = None
    for start in range(0, n, size):
        piece = ids[start : start + size]
        out = model(input_ids=piece[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1]
    return [(lyr.keys, lyr.values) for lyr in cache.layers], logits


@torch.no_grad()
def greedy_tokens(model, ids: torch.Tensor, gen_tokens: int, chunks: int = 1) -> list[int]:
    """The teacher's greedy continuation of ``ids``, optionally via a chunked prefill."""
    kv, logits = _prefill(model, ids) if chunks <= 1 else _prefill_chunked(model, ids, chunks)
    cache = GradShiftCache([(k, v) for k, v in kv])
    toks: list[int] = []
    for i in range(gen_tokens):
        toks.append(int(logits.argmax()))
        if i == gen_tokens - 1:
            break
        seen = cache.get_seq_length()
        logits = _forward_at(
            model,
            cache,
            torch.tensor([toks[-1]], device=ids.device),
            torch.tensor([seen], device=ids.device),
            seen + 1,
            1,
            scatter=False,
        )[-1]
    del cache, kv
    return toks


@torch.no_grad()
def reference_is_stable(model, ex: Example, forced: list[int], chunks: int = 2) -> bool:
    """Whether the teacher's own greedy continuation survives a benign path perturbation.

    The gate compares the student against *one* full recompute of the edited context. That
    is only a meaningful reference if the full recompute itself is determined: if two
    mathematically identical recomputes of the same text disagree about the next token, the
    model is at a near-tie there, and the resulting KL says which arbitrary branch the
    reference happened to take — not whether stitched KV changed the model's behaviour.

    Measured on iteration 3's data, this is not a hypothetical: 33 of 120 items changed
    value between the ``--base-only`` and the full-eval code paths, and those items carry
    47% of the governing bucket's total KL. Two runs of the *same* path agree on 120/120,
    so this is path-dependence rather than run-to-run randomness — which is exactly why
    averaging repeated identical passes buys nothing and a perturbation probe is needed.

    ``forced`` is the teacher's greedy continuation, which the caller already has. Given it,
    the probe costs one prefill and one *batched* teacher-forced pass rather than a second
    31-step greedy decode: if the perturbed run's argmax equals ``forced`` at every
    position, its free-running greedy decode would have produced ``forced`` exactly. Same
    question, ~17% of an eval item instead of ~33%.
    """
    ids = torch.cat([ex.new_ids, ex.query_ids])
    seq = _clean_sequence(model, ids, forced, chunks=chunks)
    return seq.argmax(-1).tolist() == list(forced)


def _clean_sequence(model, ids: torch.Tensor, forced: list[int], chunks: int = 1) -> torch.Tensor:
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
        kv, first = _prefill(model, ids) if chunks <= 1 else _prefill_chunked(model, ids, chunks)
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
    segments = example_segments(ex)
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


def example_segments(ex: Example) -> list:
    """The reuse plan for one item: explicit segments, or the single span expanded."""
    return ex.span if isinstance(ex.span, list) else span_segments(ex.span)


def clean_logits(model, ex: Example, forced: list[int]) -> torch.Tensor:
    """Student logits over the same continuation from a *clean* full-recompute context.

    The anchor term. With the adapter at identity this is the teacher, bit for bit — same
    function, same weights — so its KL is exactly 0 and any nonzero value is adapter-induced
    drift on an ordinary context, in the same units as the stitched KL it is traded against.
    """
    return _clean_sequence(model, torch.cat([ex.new_ids, ex.query_ids]), forced)


#: WSL2 does not surface GPU exhaustion as a clean ``torch.OutOfMemoryError``. Its GPU
#: paravirtualisation layer fails the residency ioctl with ``-ENOMEM``
#: (``dmesg``: ``dxgk: dxgkio_make_resident: Ioctl failed: -12``) and torch reports it as a
#: plain ``RuntimeError: CUDA driver error: device not ready``. Iteration 1 lost a run to it
#: 81 items in, and iteration 3 lost one to it again on a 8.5k-token item — the second time
#: *through* an OOM fallback that only caught the clean exception. Matching the message is
#: ugly, and it is still better than the alternative, which is that one long item ends a
#: two-hour job. The fallback is attempted once; if the retry also fails the error escapes,
#: so a genuinely poisoned context still ends the run rather than looping.
_OOM_SIGNATURES = ("out of memory", "device not ready", "make_resident")


def _is_oom(exc: BaseException) -> bool:
    """Whether an exception is GPU exhaustion, including WSL's mislabelled form."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return any(sig in str(exc).lower() for sig in _OOM_SIGNATURES)


def kv_bytes_per_token(model) -> int:
    """Bytes of K+V cache one token costs across every layer, at the model's dtype."""
    cfg = model.config
    heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    element = torch.finfo(model.dtype).bits // 8
    return 2 * cfg.num_hidden_layers * n_kv * head_dim * element


def stitch_memory_estimate(model, total_tokens: int, grad_prefill: bool) -> dict:
    """Estimate the dominant (full-length K/V) term of one training item's peak.

    Not a substitute for a measurement — ``train`` reports the real
    ``torch.cuda.max_memory_allocated`` — but it is what decides whether
    ``--grad-prefill`` can be given a cap that covers an 8k session before a run is
    launched, rather than after a two-hour job dies. Only full-length copies are counted;
    the fresh span's activations are over ``E'`` plus the query and are small beside them.

    Live at once, per item:

    * ``old_kv``  — the ``no_grad`` prefill of the old context (detached, one copy)
    * the stitched cache :func:`grad_stitch` places (constant: the reused KV is a constant)
    * with ``grad_prefill``, the ``index_copy`` result that SDPA saves for backward (+1)

    The teacher's own prefill is freed before the student runs, so it does not add a copy.
    """
    per = kv_bytes_per_token(model)
    copies = 3 if grad_prefill else 2
    return {
        "kv_bytes_per_token": per,
        "copies": copies,
        "tokens": total_tokens,
        "cache_bytes": per * total_tokens * copies,
        "cache_gib": per * total_tokens * copies / 2**30,
    }


def _peak_gib() -> float:
    """Peak allocated GiB since the last reset, or 0 off CUDA."""
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def is_governing(ex: Example) -> bool:
    """Whether this example edits a span that governs later generation."""
    return "governing" in ex.edit_kind


#: Kinds the adapter is asked to *improve* rather than merely not to damage. Governing
#: edits are the original target; ``paged`` joins them because the whole reason that
#: population exists is that it is the failure being repaired. Getting this wrong is not a
#: cosmetic mistake -- with ``--preserve-weight`` set, anything outside this set is trained
#: with the do-no-harm hinge, which is zero as long as the adapter is no worse than the
#: base, so a paged item outside it would contribute no improvement gradient at all.
TARGET_KINDS = ("paged",)


def is_target(ex: Example) -> bool:
    """Whether the loss asks this example to get *better* (vs merely not worse)."""
    return is_governing(ex) or ex.edit_kind in TARGET_KINDS


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
    guard = preserve_weight > 0 and not is_target(ex)
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

        def _forward_backward(use_gp: bool):
            """Forward, loss, and (when training) backward as one retryable unit.

            The backward has to be inside the retry, not after it: it is the memory *peak*,
            not the forward. The 2026-08-20 mixed retrain died at
            ``example_losses -> Tensor.backward`` with WSL's ENOMEM while an OOM handler
            that only wrapped the forward looked on.
            """
            stitch = kl_to(teacher_seq, stitched_logits(model, ex, old_kv, forced, use_gp))
            # governing/paged: get better. guarded kinds: just do not get worse than the base.
            t = (
                torch.clamp(stitch - (base_kl + preserve_slack), min=0.0) * preserve_weight
                if guard
                else stitch
            )
            if backward is not None:
                if t.requires_grad:
                    (t * backward).backward()
                t, stitch = float(t.detach()), float(stitch.detach())
            return stitch, t

        # A backward that dies part-way has already accumulated part of this item's
        # gradient. Retrying on top of that would count the surviving fraction twice, so the
        # adapter's grads are snapshotted and restored before the retry. Only worth doing
        # when there is something to retry *to*.
        snapshot = (
            [(lo.lora_a.grad, lo.lora_b.grad) for lo in loras]
            if backward is not None and used_grad_prefill
            else None
        )
        if snapshot is not None:
            snapshot = [
                (None if a is None else a.clone(), None if b is None else b.clone())
                for a, b in snapshot
            ]
        oom = False
        try:
            stitch_kl, term = _forward_backward(used_grad_prefill)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # A token cap is a *prediction* about memory; this is the measurement. An item
            # whose expressive path does not fit is worth training on the cheap path rather
            # than losing the whole run.
            if not _is_oom(exc) or not used_grad_prefill:
                raise
            oom, used_grad_prefill = True, False
            for lo, (a, b) in zip(loras, snapshot or [], strict=False):
                lo.lora_a.grad, lo.lora_b.grad = a, b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stitch_kl, term = _forward_backward(False)
        snapshot, old_kv = None, None

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
        "grad_prefill_oom": oom,
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
                "grad_prefill_oom": parts["grad_prefill_oom"],
                "tokens": parts["tokens"],
                "peak_gib": _peak_gib(),
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
    ref_stability: bool = False,
    on_row=None,
) -> list[dict]:
    """Per-example rows: reuse-all KL and clean-context KL, base vs adapted, plus accuracy.

    Four numbers per example, all against the *base* model's full recompute of the edited
    context — the same reference ``kvshift_eval`` uses:

        base_stitch_kl    the Phase 1 failure being repaired
        tuned_stitch_kl   the same run with the adapter on
        tuned_clean_kl    damage: adapter on, clean cache (base's is 0 by construction)
        *_answer_ok       planted-fact answer contains the expected code (when there is one)

    ``ref_stability`` adds one prefill and one teacher-forced pass per item to record
    ``ref_stable`` — see :func:`reference_is_stable`. It costs ~17% more eval time and it is
    what makes the gated statistic well posed, so gate runs set it and smoke tests do not.

    ``on_row`` is called with each row as it is produced. The gate run is ~1,200 items at
    ~5 s each, and writing the file only at the end would make an hour and a half of GPU
    time an all-or-nothing bet on nothing going wrong at item 1,199.
    """
    rows = []
    t_start = time.perf_counter()
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

        with adapters(loras, False):
            stable = reference_is_stable(model, ex, forced) if ref_stability else None

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
                # what the loss asked of this item, which is what the checkpoint rule must
                # bucket on -- a paged item is a target, not part of the do-no-harm set
                "target": is_target(ex),
                "ref_stable": stable,
                "base_stitch_kl": float(kl_to(teacher_seq, base_seq)),
                "tuned_stitch_kl": float(kl_to(teacher_seq, tuned_seq)),
                "tuned_clean_kl": float(kl_to(teacher_seq, tuned_clean)),
                "ref_answer_ok": ok(teacher_seq),
                "base_answer_ok": ok(base_seq),
                "tuned_answer_ok": ok(tuned_seq),
            }
        )
        if on_row:
            on_row(rows[-1])
        if len(rows) % 25 == 0:
            done, total = len(rows), len(examples)
            rate = (time.perf_counter() - t_start) / done
            print(
                f"  eval {done}/{total} ({rate:.1f} s/item, "
                f"~{rate * (total - done) / 60:.0f} min left)",
                flush=True,
            )
            if ex.old_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------- gated statistics
#
# Iteration 3 established that the ratio-of-means the first three iterations gated on is
# not a measurable quantity at this sample size. Two facts, both from that run's data:
#
#   * The base measurement is *exactly* reproducible for a fixed code path (two separate
#     full evals agree on 120/120 items) and *not* reproducible across paths (the
#     ``--base-only`` pre-check disagrees on 33/120). So the noise is path-dependent, not
#     stochastic, and averaging k repeats of one path returns k identical numbers.
#   * 47% of the governing bucket's total KL sits on the 12 items (of 46) whose reference
#     moves under a benign perturbation. A ratio of two means each dominated by such items
#     cannot resolve the differences the gates argue over.
#
# The replacement is the paired per-item delta: for one item, in one pass, against one
# teacher, ``tuned - base``. Everything that makes the absolute level unreliable — which
# branch of a near-tie the teacher took, which kernel cuBLAS picked — is shared by both
# columns and cancels. What is left is sampling uncertainty over items, which is what the
# bootstrap interval reports and what more items would actually reduce.

PAIRED_RESAMPLES = 10_000


def trimmed_mean(values: list[float], trim: float = 0.2) -> float:
    """Mean after dropping ``trim`` of the mass from each end (0.2 -> the 20% trimmed mean)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("trimmed_mean of an empty sample")
    k = int(len(ordered) * trim)
    kept = ordered[k : len(ordered) - k] or ordered
    return statistics.fmean(kept)


def bootstrap_ci(
    values: list[float],
    resamples: int = PAIRED_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 20260820,
    statistic=statistics.fmean,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic`` over ``values``.

    Seeded, so a reported interval is reproducible from the same rows. The percentile
    method is used rather than BCa deliberately: with n≈46 and a heavy right tail the
    acceleration term is itself poorly estimated, and a slightly conservative interval is
    the right failure mode for a gate.
    """
    if not values:
        raise ValueError("bootstrap_ci of an empty sample")
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    lo = stats[max(0, int((alpha / 2) * resamples) - 1)]
    hi = stats[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (float(lo), float(hi))


def paired_deltas(rows: list[dict], base="base_stitch_kl", tuned="tuned_stitch_kl") -> list[float]:
    """Per-item ``tuned - base``, both measured in the same pass against the same teacher."""
    return [r[tuned] - r[base] for r in rows]


def delta_summary(
    rows: list[dict],
    base="base_stitch_kl",
    tuned="tuned_stitch_kl",
    resamples: int = PAIRED_RESAMPLES,
) -> dict:
    """The gated statistic for one bucket: mean paired delta with a bootstrap interval.

    ``improved`` is a sign test that needs no distributional assumption at all, and is
    reported next to the mean because a mean delta can be carried by one item while the
    sign test says half the population got worse.
    """
    deltas = paired_deltas(rows, base, tuned)
    if not deltas:
        return {"n": 0}
    lo, hi = bootstrap_ci(deltas, resamples)
    return {
        "n": len(deltas),
        "mean": statistics.fmean(deltas),
        "median": statistics.median(deltas),
        "trimmed": trimmed_mean(deltas),
        "ci_lo": lo,
        "ci_hi": hi,
        "improved": sum(d < 0 for d in deltas),
        "significant": hi < 0 or lo > 0,
    }


def stable_rows(rows: list[dict]) -> list[dict]:
    """Rows whose reference survived the perturbation probe (or that were never probed)."""
    return [r for r in rows if r.get("ref_stable") is not False]


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


#: Pre-registered checkpoint-selection rule (docs/phase3-design.md, iteration 3). Written
#: down *before* the run so "pick the best checkpoint" cannot become "pick the checkpoint
#: that passes": governing tail first, but only among checkpoints that have not spent the
#: collateral, and the two constraints are the exit criteria's own numbers (criterion 3's
#: 20% non-governing allowance, criterion 2's 0.002 clean-drift budget).
CKPT_NON_GOV_FACTOR = 1.2
CKPT_CLEAN_MAX = 0.002


def select_checkpoint(
    history: list[dict],
    non_gov_factor: float = CKPT_NON_GOV_FACTOR,
    clean_max: float = CKPT_CLEAN_MAX,
) -> dict | None:
    """Best governing p95 among checkpoints that keep the collateral inside budget.

    Feasible means, on the mid-training held-out slice: non-governing **median** at most
    ``non_gov_factor`` x the base's median on the same items, and clean drift at most
    ``clean_max``. Among those, minimise governing p95 — the tail is what forces
    ``reuse_plan`` to refuse, so it is the quantity the phase is buying.

    Returns ``None`` when no checkpoint is feasible, which is a reportable outcome (the run
    bought the tail only by spending the collateral at every point measured) and not an
    error to paper over by relaxing the rule after the fact.
    """
    feasible = [
        h
        for h in history
        if h.get("gov_p95") is not None
        and h.get("non_tuned_median") is not None
        and h["non_tuned_median"] <= h.get("non_base_median", 0.0) * non_gov_factor
        and h.get("clean", float("inf")) <= clean_max
    ]
    return min(feasible, key=lambda h: h["gov_p95"]) if feasible else None


def report(rows: list[dict]) -> str:
    """The before/after table. Governing vs non-governing is the whole question."""
    # ``standing-governing`` is a governing kind, but it is bucketed separately as well as
    # inside ``governing``: it was added after the gate ratios were pre-registered, so the
    # headline ratio stays on the same core population (``governing`` + ``mid-governing``)
    # that iteration 1 and 2 measured, and the new bucket is reported next to it rather than
    # silently folded into it.
    standing = [r for r in rows if r["edit_kind"] == STANDING_KIND]
    core = [r for r in rows if r["governing"] and r["edit_kind"] != STANDING_KIND]
    non = [r for r in rows if not r["governing"]]
    buckets = [
        ("governing", core),
        ("standing-gov", standing),
        ("governing+std", core + standing if standing else []),
        ("non-governing", non),
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
    # both statistics, because they say different things: the 144-session eval's headline
    # 9x is a ratio of means (driven by the heavy tail governing owns), while the median
    # ratio is ~3-4.5x and describes the typical item. A fix must move both.
    for label, gov in (("", core), ("+std ", core + standing)):
        if not (gov and non) or (label and not standing):
            continue
        for tag in ("base", "tuned"):
            for stat, fn in (("mean", statistics.fmean), ("median", statistics.median)):
                g = fn([r[f"{tag}_stitch_kl"] for r in gov])
                n = fn([r[f"{tag}_stitch_kl"] for r in non])
                out.append(
                    f"{tag:<16}{label}governing/non-governing {stat} KL ratio "
                    f"= {g / max(n, 1e-9):.2f}x  ({g:.4f} / {n:.4f})   [descriptive only]"
                )

    # --- the gated statistic ---------------------------------------------------------
    probed = [r for r in rows if r.get("ref_stable") is not None]
    if probed:
        unstable = [r for r in probed if not r["ref_stable"]]
        share = sum(r["base_stitch_kl"] for r in unstable) / max(
            sum(r["base_stitch_kl"] for r in probed), 1e-12
        )
        out.append(
            f"\nreference stability: {len(probed) - len(unstable)}/{len(probed)} stable; "
            f"the {len(unstable)} unstable carry {100 * share:.0f}% of total base KL "
            f"(excluded from the gate, reported here)"
        )
    out.append(
        f"\n{'bucket (stable only)':<24}{'n':>4}{'mean delta':>12}{'95% CI':>24}"
        f"{'median':>10}{'trim20':>10}{'improved':>10}"
    )
    for name, rs in buckets:
        rs = stable_rows(rs)
        if not rs:
            continue
        d = delta_summary(rs)
        ci = f"[{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]"
        imp = f"{d['improved']}/{d['n']}"
        out.append(
            f"{name:<24}{d['n']:>4}{d['mean']:>+12.5f}{ci:>24}"
            f"{d['median']:>+10.5f}{d['trimmed']:>+10.5f}{imp:>10}"
        )
    return "\n".join(out)


# ------------------------------------------------------- free-running discriminator


@torch.no_grad()
def stitched_greedy(model, ex: Example, old_kv, gen_tokens: int) -> list[int]:
    """Free-running greedy continuation read off the **stitched** cache.

    :func:`stitched_logits` teacher-forces the reference's own continuation, which is the
    right thing for a KL and the wrong thing for "would the served answer differ": a
    teacher-forced pass is re-anchored to the reference at every step, so a divergence that
    would compound over 32 free-running tokens is corrected before it can. This is the
    serving decode instead — each token is chosen by the model and fed back to itself.
    """
    all_ids = torch.cat([ex.new_ids, ex.query_ids])
    total = int(all_ids.shape[0])
    segments = example_segments(ex)
    cache = grad_stitch(old_kv, segments, total, inv_freq_of(model))
    positions = fresh_positions(segments, total, all_ids.device)
    logits = _forward_at(model, cache, all_ids[positions], positions, total, 1)[-1]
    toks: list[int] = []
    for i in range(gen_tokens):
        toks.append(int(logits.argmax()))
        if i == gen_tokens - 1:
            break
        seen = cache.get_seq_length()
        logits = _forward_at(
            model,
            cache,
            torch.tensor([toks[-1]], device=all_ids.device),
            torch.tensor([seen], device=all_ids.device),
            seen + 1,
            1,
            scatter=False,
        )[-1]
    del cache
    return toks


@torch.no_grad()
def free_run(
    model,
    loras: list[LoRALinear],
    examples: list[Example],
    tok,
    gen_tokens: int = 32,
    with_tuned: bool = False,
    on_row=None,
) -> list[dict]:
    """Serving-shaped comparison: free-running greedy, stitched cache vs full recompute.

    This is the discriminator between two explanations of the composition failure. Track L
    measures free-running exact match on the real workload and sees it collapse (fact EM
    7/14); the paged gate run measures 32 teacher-forced tokens and sees planted-fact
    survive 498/500 at a KL median of 0.0186. Either teacher forcing was hiding an
    answer-level failure, or the failure lives in the serving path rather than in stitched
    attention. Same population, same stitched cache, free-running decode: if EM collapses
    here, it is the former; if it holds, the connector is implicated.
    """
    rows = []
    for ex in examples:
        with adapters(loras, False):
            ref = greedy_tokens(model, torch.cat([ex.new_ids, ex.query_ids]), gen_tokens)
            old_kv, _ = _prefill(model, ex.old_ids)
            old_kv = [(k.detach(), v.detach()) for k, v in old_kv]
            got = stitched_greedy(model, ex, old_kv, gen_tokens)
        tuned = None
        if with_tuned:
            with adapters(loras, True):
                t_kv, _ = _prefill(model, ex.old_ids)
                tuned = stitched_greedy(
                    model, ex, [(k.detach(), v.detach()) for k, v in t_kv], gen_tokens
                )
                del t_kv

        def hit(toks, expected=ex.expected):
            if not expected or tok is None:
                return None
            text = tok.decode(toks)
            return any(e in text for e in expected)

        rows.append(
            {
                "sid": ex.sid,
                "edit_kind": ex.edit_kind,
                "qtype": ex.qtype,
                "exact_match": got == ref,
                "tuned_exact_match": None if tuned is None else tuned == ref,
                # how far in the two decodes agree: 32 means identical, 0 means the very
                # first served token differs
                "agree_prefix": next(
                    (i for i, (a, b) in enumerate(zip(ref, got, strict=False)) if a != b),
                    min(len(ref), len(got)),
                ),
                "ref_fact_ok": hit(ref),
                "base_fact_ok": hit(got),
                "tuned_fact_ok": None if tuned is None else hit(tuned),
            }
        )
        if on_row:
            on_row(rows[-1])
        if len(rows) % 25 == 0:
            print(f"  freerun {len(rows)}/{len(examples)}", flush=True)
            if ex.old_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def free_run_report(rows: list[dict]) -> str:
    """Exact match and planted-fact accuracy — the statistics serving is judged on."""
    n = len(rows)
    if not n:
        return "no rows"
    em = sum(bool(r["exact_match"]) for r in rows)
    graded = [r for r in rows if r["ref_fact_ok"] is not None]
    out = [
        f"free-running greedy, {n} items, stitched cache vs full recompute",
        f"  exact match (32 tokens)   {em}/{n} = {em / n:.3f}",
        f"  mean agreeing prefix      {statistics.fmean(r['agree_prefix'] for r in rows):.1f}"
        f" / 32 tokens   (median {statistics.median(r['agree_prefix'] for r in rows):.0f})",
        f"  first served token differs {sum(r['agree_prefix'] == 0 for r in rows)}/{n}",
    ]
    if graded:
        rf = sum(bool(r["ref_fact_ok"]) for r in graded)
        bf = sum(bool(r["base_fact_ok"]) for r in graded)
        out.append(
            f"  planted-fact EM           reference {rf}/{len(graded)} = {rf / len(graded):.3f}"
            f" ; stitched {bf}/{len(graded)} = {bf / len(graded):.3f}"
        )
        lost = sum(1 for r in graded if r["ref_fact_ok"] and not r["base_fact_ok"])
        gained = sum(1 for r in graded if r["base_fact_ok"] and not r["ref_fact_ok"])
        out.append(f"  facts lost to reuse {lost}, gained {gained}")
    tuned = [r for r in rows if r["tuned_exact_match"] is not None]
    if tuned:
        tem = sum(bool(r["tuned_exact_match"]) for r in tuned)
        tg = [r for r in tuned if r["tuned_fact_ok"] is not None]
        line = f"  ADAPTER exact match       {tem}/{len(tuned)} = {tem / len(tuned):.3f}"
        if tg:
            tf = sum(bool(r["tuned_fact_ok"]) for r in tg)
            line += f" ; planted-fact {tf}/{len(tg)} = {tf / len(tg):.3f}"
        out.append(line)
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
    ap.add_argument("cmd", choices=["train", "eval", "probe", "freerun"])
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
    ap.add_argument(
        "--population",
        default="synthetic",
        choices=["synthetic", "paged", "mixed"],
        help="synthetic = kvshift_eval single-span edits; paged = cold-tier demote/promote "
        "transitions (marathon.paged_eval), the shape Track L's failing workload has; "
        "mixed = half and half",
    )
    ap.add_argument("--paged-turns", type=int, default=40)
    ap.add_argument("--paged-active-tokens", type=int, default=6000)
    ap.add_argument("--base-only", action="store_true", help="skip the tuned columns")
    ap.add_argument(
        "--ref-stability",
        action="store_true",
        help="probe whether each item's teacher survives a benign path perturbation; "
        "unstable items are excluded from the gated statistic (one extra recompute/item)",
    )
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
        "--standing-frac",
        type=float,
        default=0.0,
        help="share of the governing half drawn as probe-shaped standing-instruction "
        "sessions (closes the dep-instruction distribution gap; own bucket in the report)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save the adapter and run a held-out mid-training eval every N items",
    )
    ap.add_argument(
        "--mid-eval-items", type=int, default=24, help="sessions in each mid-training eval"
    )
    ap.add_argument(
        "--mid-eval-seed",
        type=int,
        default=None,
        help="seed for the mid-training slice the checkpoint rule selects on. MUST differ "
        "from the eval seed: it defaulted to --seed+2000, which for the standard "
        "--seed 7001 training run is 9001 -- the eval seed -- so iteration 3 selected its "
        "checkpoint using 24 items that were also in its 120-item held-out eval.",
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

    def _synthetic(n, seed):
        return build_examples(
            tok,
            dev,
            n,
            seed,
            args.gov_frac,
            args.min_tokens,
            args.max_tokens,
            args.queries_per_item,
            standing_frac=args.standing_frac,
        )

    def _paged(n, seed):
        from .paged_eval import build_paged_examples

        return build_paged_examples(tok, dev, n, seed, args.paged_turns, args.paged_active_tokens)

    if args.population == "paged":
        examples = _paged(args.items, args.seed)
    elif args.population == "mixed":
        half = args.items // 2
        examples = _synthetic(args.items - half, args.seed) + _paged(half, args.seed)
        random.Random(args.seed).shuffle(examples)
    else:
        examples = _synthetic(args.items, args.seed)
    print(
        f"{args.cmd}: {len(examples)} examples from {args.items} sessions "
        f"(seed {args.seed}, population {args.population}, gov_frac {args.gov_frac}), "
        f"model {args.model}",
        flush=True,
    )
    paged = [e for e in examples if e.edit_kind == "paged"]
    if paged:
        seg = statistics.fmean(e.meta["segments"] for e in paged)
        reuse = statistics.fmean(e.meta["reused"] / e.meta["new_tokens"] for e in paged)
        print(
            f"  paged: {len(paged)} items, {seg:.1f} reuse segments/item, "
            f"{100 * reuse:.0f}% of the new view reused, "
            f"{statistics.fmean(e.meta['new_tokens'] for e in paged):.0f} tokens/view",
            flush=True,
        )

    if args.cmd == "freerun":
        with contextlib.ExitStack() as stack:
            sink = (
                stack.enter_context(open(args.jsonl, "w", encoding="utf-8")) if args.jsonl else None
            )

            def emit(row):
                sink.write(json.dumps(row) + "\n")
                sink.flush()

            rows = free_run(
                model,
                loras,
                examples,
                tok,
                args.gen_tokens,
                with_tuned=bool(args.lora),
                on_row=emit if sink else None,
            )
        print("\n" + free_run_report(rows))
        return 0

    if args.cmd == "train":
        if args.grad_prefill:
            longest = max(int(ex.new_ids.shape[0] + ex.query_ids.shape[0]) for ex in examples)
            est = stitch_memory_estimate(model, longest, True)
            over = sum(
                1
                for ex in examples
                if args.grad_prefill_max_tokens > 0
                and int(ex.new_ids.shape[0] + ex.query_ids.shape[0]) > args.grad_prefill_max_tokens
            )
            print(
                f"grad-prefill: longest item {longest} tokens, estimated full-length K/V "
                f"{est['cache_gib']:.2f} GiB ({est['copies']} copies at "
                f"{est['kv_bytes_per_token'] / 1024:.0f} KiB/token); "
                f"{over}/{len(examples)} items over the {args.grad_prefill_max_tokens}-token "
                f"cap will use the cheap path",
                flush=True,
            )
        mid: list[Example] = []
        if args.checkpoint_every and args.mid_eval_items:
            # a *held-out* slice: same seed as the final eval, so the mid-training curve and
            # the final number are measured on the same population
            mid = build_examples(
                tok,
                dev,
                args.mid_eval_items,
                args.mid_eval_seed if args.mid_eval_seed is not None else args.seed + 2000,
                args.gov_frac,
                args.min_tokens,
                args.max_tokens,
                args.queries_per_item,
                standing_frac=args.standing_frac,
            )
            print(f"mid-training eval slice: {len(mid)} held-out examples", flush=True)

        history: list[dict] = []

        def on_checkpoint(step, _log):
            path = f"{args.out}.step{step}" if args.out else None
            if path:
                torch.save(lora_state(loras), path)
            row = {"step": step, "path": path, "peak_gib": _peak_gib()}
            if mid:
                rows = evaluate(model, loras, mid, tok, args.gen_tokens)
                g = [r for r in rows if r.get("target", r["governing"])]
                n = [r for r in rows if not r.get("target", r["governing"])]
                # p95 and the medians, not just the means: `select_checkpoint` needs the
                # governing tail and the non-governing typical item, which are exactly the
                # two statistics the exit criteria are written on.
                for tag, rs in (("gov", g), ("non", n)):
                    if not rs:
                        continue
                    base, tuned = _agg(rs, "base_stitch_kl"), _agg(rs, "tuned_stitch_kl")
                    row[f"{tag}_base"] = base["mean"]
                    row[f"{tag}_tuned"] = tuned["mean"]
                    row[f"{tag}_base_median"] = base["median"]
                    row[f"{tag}_tuned_median"] = tuned["median"]
                    row[f"{tag}_p95"] = tuned["p95"]
                row["clean"] = statistics.fmean([r["tuned_clean_kl"] for r in rows])
                print(
                    f"[checkpoint {step}] gov {row.get('gov_base', float('nan')):.4f}->"
                    f"{row.get('gov_tuned', float('nan')):.4f} (p95 "
                    f"{row.get('gov_p95', float('nan')):.4f})  non "
                    f"{row.get('non_base', float('nan')):.4f}->"
                    f"{row.get('non_tuned', float('nan')):.4f} (median "
                    f"{row.get('non_base_median', float('nan')):.4f}->"
                    f"{row.get('non_tuned_median', float('nan')):.4f})  "
                    f"clean {row['clean']:.4f}  peak {row['peak_gib']:.1f} GiB",
                    flush=True,
                )
            history.append(row)
            emit_ckpt(row)

        # Stream the training rows and the checkpoint curve as they happen. Yesterday's
        # mixed retrain died at item ~20 of 200 and left nothing behind at all, because both
        # files were written after the loop: an hour of GPU time with no curve to look at.
        with contextlib.ExitStack() as stack:
            train_sink = (
                stack.enter_context(open(args.jsonl, "w", encoding="utf-8")) if args.jsonl else None
            )
            ckpt_sink = (
                stack.enter_context(open(f"{args.jsonl}.checkpoints", "w", encoding="utf-8"))
                if args.jsonl
                else None
            )

            def _emit(sink):
                def go(row):
                    if sink:
                        sink.write(json.dumps(row) + "\n")
                        sink.flush()

                return go

            emit_step, emit_ckpt = _emit(train_sink), _emit(ckpt_sink)
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
                on_step=emit_step,
                anchor_every=args.anchor_every,
                accum=args.accum,
            )
        if args.out:
            torch.save(lora_state(loras), args.out)
            print(f"wrote adapter -> {args.out}")
        downgraded = sum(1 for r in log if args.grad_prefill and not r["grad_prefill"])
        if args.grad_prefill:
            ooms = sum(1 for r in log if r.get("grad_prefill_oom"))
            print(
                f"grad-prefill: {len(log) - downgraded}/{len(log)} items kept the fresh span "
                f"in the graph (cap {args.grad_prefill_max_tokens} tokens, "
                f"{ooms} fell back on OOM)"
            )
        print(f"peak allocated: {_peak_gib():.2f} GiB")
        if history:
            # the pre-registered rule, applied to the measured curve rather than by eye
            pick = select_checkpoint(history)
            print(
                f"selected checkpoint (gov p95 subject to non-gov median <= base x"
                f"{CKPT_NON_GOV_FACTOR} and clean <= {CKPT_CLEAN_MAX}): "
                + (
                    f"step {pick['step']} -> {pick['path']} "
                    f"(gov p95 {pick['gov_p95']:.4f}, non median "
                    f"{pick['non_tuned_median']:.4f} vs base "
                    f"{pick['non_base_median']:.4f}, clean {pick['clean']:.4f})"
                    if pick
                    else "NONE feasible — every checkpoint spent the collateral"
                ),
                flush=True,
            )
        gov = [r["stitch_kl"] for r in log if "governing" in r["edit_kind"]]
        if gov:
            half = len(gov) // 2
            print(
                f"governing stitch_kl: first half median {statistics.median(gov[:half]):.4f} "
                f"-> second half median {statistics.median(gov[half:]):.4f}"
            )
        return 0

    # `--base-only` was accepted but never forwarded until 2026-08-20, so the pre-check ran
    # the full path at double cost. It produced the *right* numbers anyway — with no
    # `--lora` the adapter is identity, so the tuned columns equal the base ones bit for bit
    # — which is why the bug survived three runs. It is also part of why the iteration-3
    # pre-check disagreed with the in-run base on 33/120 items: a loaded adapter changes the
    # allocation pattern, and that changes which kernels cuBLAS picks for everything after.
    with contextlib.ExitStack() as stack:
        sink = stack.enter_context(open(args.jsonl, "w", encoding="utf-8")) if args.jsonl else None

        def write_row(row):
            sink.write(json.dumps(row) + "\n")
            sink.flush()  # a killed run must keep the rows it already paid for

        rows = evaluate(
            model,
            loras,
            examples,
            tok,
            args.gen_tokens,
            args.base_only,
            args.ref_stability,
            on_row=write_row if sink else None,
        )
    print("\n" + report(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
