"""Position-shifted KV reuse: the direct attack on DESIGN.md's "positional entanglement".

The delta engine knows a turn changed one span. Turn that into KV work:

    old sequence   [ P ][ E  ][ S ]      fully cached, K rotated at its old positions
    new sequence   [ P ][ E' ][ S ]      S is byte-identical but shifted by d = |E'|-|E|

1. ``P``  reuse KV as-is (it is a prefix; nothing moved).
2. ``E'`` compute fresh, attending to P's cache.
3. ``S``  reuse V unchanged (V carries no position) and reuse K *re-rotated* by ``d``.
   RoPE is a rotation by angle ``p * inv_freq``, so a key computed at ``p`` is moved
   to ``p + d`` exactly by rotating it once more by ``d * inv_freq`` — no recompute,
   no approximation (see :func:`rerotate_keys`, unit-tested bit-close in
   ``tests/test_kvshift.py``).
4. What re-rotation cannot fix: S's KV was computed *attending to* ``E``, not ``E'``.
   That is the residual error, and it is what selective recompute buys back.
   Policies: recompute nothing, recompute the first M tokens after the edit, or
   pick the top-r fraction by K deviation at a cheap early layer (CacheBlend-style).

Everything below is plain HF transformers eager attention: a ``Cache`` whose layers
scatter freshly computed K/V into fixed slots of a pre-stitched buffer and return the
whole buffer, so a recomputed token attends to the already-stitched context around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, DynamicLayer

from .diff import Copy, compute_delta

# --------------------------------------------------------------------------- RoPE


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rerotate_keys(k: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Move RoPE'd keys ``delta`` positions: ``rerotate(K@p, d) == K@(p+d)``, exactly.

    ``k`` is ``[..., seq, head_dim]`` already carrying RoPE at its original positions.
    Rotating by a constant angle commutes with the original rotation, so no knowledge
    of the original positions is needed.
    """
    if delta == 0:
        return k
    angles = float(delta) * inv_freq.to(torch.float32)
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos().to(k.dtype), emb.sin().to(k.dtype)
    return k * cos + rotate_half(k) * sin


def inv_freq_of(model) -> torch.Tensor:
    """The model's own RoPE inverse frequencies (respects rope scaling)."""
    return model.model.rotary_emb.inv_freq.detach()


# ------------------------------------------------------------------- edit location


@dataclass(frozen=True)
class Span:
    """Where the edit is, in tokens. ``P = [:p]``, ``E/E' `` next, ``S`` the rest."""

    p: int  # unchanged prefix length (tokens)
    e_old: int  # replaced span length in the old sequence
    e_new: int  # replacement length in the new sequence
    s: int  # unchanged suffix length (tokens), identical in both

    @property
    def delta(self) -> int:
        return self.e_new - self.e_old

    @property
    def new_len(self) -> int:
        return self.p + self.e_new + self.s


def byte_span(base: bytes, target: bytes) -> tuple[int, int]:
    """(unchanged prefix bytes, unchanged suffix bytes) as the delta engine sees them."""
    ops = compute_delta(base, target).ops
    head = ops[0].length if ops and isinstance(ops[0], Copy) and ops[0].offset == 0 else 0
    # everything after the last freshly inserted byte came verbatim from the baseline
    pos = last_insert_end = 0
    for op in ops:
        pos += op.length if isinstance(op, Copy) else len(op.data)
        if not isinstance(op, Copy):
            last_insert_end = pos
    return head, len(target) - last_insert_end


def token_span(old_ids: list[int], new_ids: list[int]) -> Span:
    """Token-level edit span (snapped out from the byte delta to token boundaries)."""
    p = 0
    while p < len(old_ids) and p < len(new_ids) and old_ids[p] == new_ids[p]:
        p += 1
    s = 0
    while s < len(old_ids) - p and s < len(new_ids) - p and old_ids[-1 - s] == new_ids[-1 - s]:
        s += 1
    return Span(p=p, e_old=len(old_ids) - p - s, e_new=len(new_ids) - p - s, s=s)


# ------------------------------------------------------------------- scatter cache


class _ScatterLayer(DynamicLayer):
    """Holds a full-length K/V buffer; ``update`` writes into ``owner.index`` slots."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, box: list) -> None:
        super().__init__()
        self.keys, self.values = keys, values
        self.box = box  # a list, not the parent: a cycle here strands caches on the GPU
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True
        self.last_written: torch.Tensor | None = None

    def update(self, key_states, value_states, *args, **kwargs):
        self.last_written = key_states
        index = self.box[0]
        if index is None:  # decode: plain append
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        else:
            self.keys[:, :, index] = key_states.to(self.keys.dtype)
            self.values[:, :, index] = value_states.to(self.values.dtype)
        return self.keys, self.values


class ShiftCache(Cache):
    """A cache pre-filled with reused KV; selected slots are overwritten in place."""

    def __init__(self, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._box: list = [None]
        super().__init__(layers=[_ScatterLayer(k, v, self._box) for k, v in kv])

    @property
    def index(self) -> torch.Tensor | None:
        return self._box[0]

    @index.setter
    def index(self, value: torch.Tensor | None) -> None:
        self._box[0] = value

    def clone(self) -> ShiftCache:
        return ShiftCache([(lyr.keys.clone(), lyr.values.clone()) for lyr in self.layers])


def stitch(
    old_kv: list[tuple[torch.Tensor, torch.Tensor]],
    span: Span,
    tail: int,
    inv_freq: torch.Tensor,
    rotate: bool = True,
) -> ShiftCache:
    """Build the new cache: P verbatim, E' zeroed, S re-rotated by ``span.delta``, tail zeroed.

    ``tail`` is the number of trailing slots (the new query turn) computed fresh.
    """
    total = span.new_len + tail
    out = []
    s_from = span.p + span.e_old
    s_to = span.p + span.e_new
    for k, v in old_kv:
        nk = k.new_zeros((k.shape[0], k.shape[1], total, k.shape[3]))
        nv = v.new_zeros((v.shape[0], v.shape[1], total, v.shape[3]))
        nk[:, :, : span.p] = k[:, :, : span.p]
        nv[:, :, : span.p] = v[:, :, : span.p]
        if span.s:
            nk[:, :, s_to : s_to + span.s] = rerotate_keys(
                k[:, :, s_from : s_from + span.s], span.delta if rotate else 0, inv_freq
            )
            nv[:, :, s_to : s_to + span.s] = v[:, :, s_from : s_from + span.s]
        out.append((nk, nv))
    return ShiftCache(out)


# ------------------------------------------------------------------------ forwards


def _causal_mask(positions: torch.Tensor, total: int, dtype: torch.dtype) -> torch.Tensor:
    cols = torch.arange(total, device=positions.device)
    allowed = cols[None, :] <= positions[:, None]
    mask = torch.zeros(allowed.shape, dtype=dtype, device=positions.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


def _run(model, cache: ShiftCache, ids: torch.Tensor, positions: torch.Tensor, total: int):
    cache.index = positions
    mask = _causal_mask(positions, total, model.dtype)
    out = model(
        input_ids=ids[None],
        attention_mask=mask,
        position_ids=positions[None],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    cache.index = None
    return out.logits[0, -1]


@torch.no_grad()
def prefill(model, ids: torch.Tensor):
    """Full recompute reference: returns (kv list, last-position logits)."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
    kv = [(lyr.keys, lyr.values) for lyr in cache.layers]
    return kv, out.logits[0, -1]


@torch.no_grad()
def greedy(model, cache, logits: torch.Tensor, n: int):
    """Continue greedily from ``logits`` using ``cache``. Returns (tokens, per-step logits)."""
    toks: list[int] = []
    seq = [logits]
    seen = cache.layers[0].keys.shape[-2]
    for i in range(n):
        nxt = int(logits.argmax())
        toks.append(nxt)
        if i == n - 1:
            break
        pos = torch.tensor([seen + i], device=logits.device)
        mask = torch.zeros((1, 1, 1, seen + i + 1), dtype=model.dtype, device=logits.device)
        out = model(
            input_ids=torch.tensor([[nxt]], device=logits.device),
            attention_mask=mask,
            position_ids=pos[None],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits[0, -1]
        seq.append(logits)
    return toks, torch.stack(seq)


@torch.no_grad()
def forced_logits(model, cache, ref_tokens: list[int]) -> torch.Tensor:
    """Logits at every position of ``ref_tokens`` fed as a teacher-forced continuation.

    One batched forward, so the comparison against full recompute does not inherit the
    all-or-nothing noise of a free-running greedy decode.
    """
    n = len(ref_tokens)
    device = cache.layers[0].keys.device
    seen = cache.layers[0].keys.shape[-2]
    positions = torch.arange(seen, seen + n, device=device)
    mask = _causal_mask(positions, seen + n, model.dtype)
    ids = torch.tensor(ref_tokens, device=device)[None]
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions[None],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=n,
    )
    return out.logits[0]


# ------------------------------------------------------------------------ policies


@dataclass
class Policy:
    """How much of S to recompute. ``kind`` in {none, firstm, blend}."""

    kind: str = "none"
    m: int = 0  # firstm: tokens of S after the edit
    ratio: float = 0.0  # blend: fraction of S recomputed
    check_layer: int = 1  # blend: layer whose K deviation ranks tokens
    rerotate: bool = True  # control: False reuses S's keys at their stale angles
    name: str = field(default="", compare=False)

    def label(self) -> str:
        if not self.rerotate:
            return "no-rerotate"
        return (
            self.name
            or {
                "none": "reuse-all",
                "firstm": f"first-{self.m}",
                "blend": f"blend-r{self.ratio:g}",
            }[self.kind]
        )


@torch.no_grad()
def _deviation(model, cache: ShiftCache, ids, positions, total, span, check_layer):
    """Fresh-vs-cached K deviation at ``check_layer`` for every token of S.

    Costs ``(check_layer+1)/n_layers`` of a full prefill over the selected tokens —
    charged honestly in the reported effective fraction.
    """
    probe = cache.clone()
    probe.index = positions
    s_to = span.p + span.e_new
    s_slots = torch.arange(s_to, s_to + span.s, device=positions.device)
    cached = probe.layers[check_layer].keys[:, :, s_slots].clone()
    h = model.model.embed_tokens(ids[None])
    pos_emb = model.model.rotary_emb(h, positions[None])
    mask = _causal_mask(positions, total, model.dtype)
    for layer in model.model.layers[: check_layer + 1]:
        h = layer(
            h,
            attention_mask=mask,
            position_embeddings=pos_emb,
            position_ids=positions[None],
            past_key_values=probe,
            use_cache=True,
        )
    fresh = probe.layers[check_layer].keys[:, :, s_slots]
    return (fresh.float() - cached.float()).norm(dim=-1).mean(dim=(0, 1))  # [s]


@torch.no_grad()
def select(model, policy: Policy, cache: ShiftCache, span: Span, ids, positions, total):
    """Indices *within S* (0-based) to recompute, and the extra-cost fraction paid."""
    if policy.kind == "none" or span.s == 0:
        return torch.zeros(0, dtype=torch.long, device=ids.device), 0.0
    if policy.kind == "firstm":
        n = min(policy.m, span.s)
        return torch.arange(n, device=ids.device), 0.0
    if policy.kind == "blend":
        dev = _deviation(model, cache, ids, positions, total, span, policy.check_layer)
        n = max(1, int(round(policy.ratio * span.s)))
        idx = dev.topk(n).indices.sort().values
        return idx, (policy.check_layer + 1) / model.config.num_hidden_layers
    raise ValueError(policy.kind)


# --------------------------------------------------------------------------- driver


@torch.no_grad()
def run_policy(
    model,
    old_kv,
    span: Span,
    new_ids: torch.Tensor,
    query_ids: torch.Tensor,
    policy: Policy,
    max_new_tokens: int = 16,
    forced: list[int] | None = None,
) -> dict:
    """Stitch, selectively recompute, answer. Returns tokens, first logits and cost."""
    device = new_ids.device
    inv_freq = inv_freq_of(model)
    tail = int(query_ids.shape[0])
    total = span.new_len + tail
    all_ids = torch.cat([new_ids, query_ids])

    fresh_always = torch.cat(
        [
            torch.arange(span.p, span.p + span.e_new, device=device),
            torch.arange(span.new_len, total, device=device),
        ]
    )

    cache = stitch(old_kv, span, tail, inv_freq, rotate=policy.rerotate)
    s_to = span.p + span.e_new
    # policy selection needs a candidate forward set: everything not reusable + all of S
    probe_pos = (
        torch.cat([fresh_always, torch.arange(s_to, s_to + span.s, device=device)]).sort().values
    )
    picked, extra = select(model, policy, cache, span, all_ids[probe_pos], probe_pos, total)
    positions = torch.cat([fresh_always, picked + s_to]).sort().values

    import time

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    logits = _run(model, cache, all_ids[positions], positions, total)
    sync()
    prefill_s = time.perf_counter() - t0
    seq = logits[None]
    if forced and len(forced) > 1:
        # teacher-force the reference continuation on a copy, for per-position metrics
        seq = torch.cat([logits[None], forced_logits(model, cache.clone(), forced[:-1])], 0)
    toks, _ = greedy(model, cache, logits, max_new_tokens)
    sync()
    wall = time.perf_counter() - t0

    recomputed = int(positions.shape[0])
    return {
        "policy": policy.label(),
        "recomputed_tokens": recomputed,
        "recompute_frac": recomputed / total,
        "effective_frac": recomputed / total + extra,
        "prefill_s": prefill_s,
        "wall_s": wall,
        "tokens": toks,
        "logits": logits,
        "logits_seq": seq,
    }


@torch.no_grad()
def run_full(model, new_ids: torch.Tensor, query_ids: torch.Tensor, max_new_tokens: int = 16):
    """Full recompute of the whole new sequence — the correctness reference."""
    import time

    ids = torch.cat([new_ids, query_ids])
    device = ids.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    from transformers.cache_utils import DynamicCache

    sync()
    t0 = time.perf_counter()
    cache = DynamicCache(config=model.config)
    logits = model(
        input_ids=ids[None], past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits[0, -1]
    sync()
    prefill_s = time.perf_counter() - t0
    toks, seq = greedy(model, cache, logits, max_new_tokens)
    sync()
    return {
        "policy": "full-recompute",
        "recomputed_tokens": int(ids.shape[0]),
        "recompute_frac": 1.0,
        "effective_frac": 1.0,
        "prefill_s": prefill_s,
        "wall_s": time.perf_counter() - t0,
        "tokens": toks,
        "logits": logits,
        "logits_seq": seq,
    }


def compare(ref: dict, got: dict) -> dict:
    """Quality of a stitched run against the full-recompute reference."""
    p = torch.softmax(ref["logits"].float(), dim=-1)
    q = torch.log_softmax(got["logits"].float(), dim=-1)
    kl = float((p * (p.clamp_min(1e-12).log() - q)).sum())
    agree = 0
    for a, b in zip(ref["tokens"], got["tokens"], strict=False):
        if a != b:
            break
        agree += 1
    n = min(ref["logits_seq"].shape[0], got["logits_seq"].shape[0])
    rp = torch.softmax(ref["logits_seq"][:n].float(), dim=-1)
    rq = torch.log_softmax(got["logits_seq"][:n].float(), dim=-1)
    kl_seq = (rp * (rp.clamp_min(1e-12).log() - rq)).sum(-1)
    tf_top1 = float(
        (ref["logits_seq"][:n].argmax(-1) == got["logits_seq"][:n].argmax(-1)).float().mean()
    )
    return {
        "kl_first": kl,
        "kl_mean_forced": float(kl_seq.mean()),
        "kl_max_forced": float(kl_seq.max()),
        "tf_top1_agree": tf_top1,
        "top1_match": ref["tokens"][:1] == got["tokens"][:1],
        "greedy_prefix_agree": agree / max(len(ref["tokens"]), 1),
        "max_logit_diff": float((ref["logits"].float() - got["logits"].float()).abs().max()),
    }
