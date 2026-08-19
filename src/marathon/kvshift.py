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

Multi-span and moved blocks are the same mechanism with the loop pulled out one level:
:func:`token_segments` asks ``marathon.diff``'s rsync matcher which runs of tokens
survived, each survivor becomes a :class:`Segment` with its *own* ``delta = dst - src``
(negative when a block moved earlier), and :func:`stitch_segments` places them all. Only
the genuinely new tokens are left over, and they are computed in one masked forward, so
each fresh span sees every stitched and freshly written slot below it.

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


# --------------------------------------------------------------- multi-span edits


@dataclass(frozen=True)
class Segment:
    """One reused run of tokens: ``old[src_start:src_end]`` lands at ``dst_start``.

    ``delta`` is per segment, so a *moved* block is simply a segment whose delta
    differs from its neighbours' (and may be negative). Segments are kept in
    destination order; everything they do not cover is recomputed.
    """

    src_start: int
    src_end: int
    dst_start: int

    @property
    def length(self) -> int:
        return self.src_end - self.src_start

    @property
    def dst_end(self) -> int:
        return self.dst_start + self.length

    @property
    def delta(self) -> int:
        return self.dst_start - self.src_start


def span_segments(span: Span) -> list[Segment]:
    """The single-edit :class:`Span` as segments: ``P`` at delta 0, then ``S`` at delta."""
    segs = [Segment(0, span.p, 0)] if span.p else []
    if span.s:
        src = span.p + span.e_old
        segs.append(Segment(src, src + span.s, span.p + span.e_new))
    return segs


_W = 4  # bytes per token id in the id-stream encoding below


def _id_bytes(ids: list[int]) -> bytes:
    return b"".join(int(i).to_bytes(_W, "big") for i in ids)


def token_segments(
    old_ids: list[int],
    new_ids: list[int],
    block_tokens: int = 16,
    min_tokens: int = 16,
) -> list[Segment]:
    """Reusable segments between two token sequences, via ``marathon.diff``.

    The rsync matcher is the right engine here precisely because it is *not* an LCS:
    it indexes every aligned baseline block in a hash table and matches wherever the
    bytes occur, so a moved block is found with a copy whose source offset runs
    backwards. Token ids are encoded as fixed 4-byte words, which makes every match
    snap to a token boundary for free (an unaligned match is a cross-token
    coincidence and is dropped). Matches shorter than ``min_tokens`` are dropped too:
    a handful of reused tokens is not worth a segment.

    Byte identity is not context identity — a segment's KV was computed after whatever
    preceded it in the *old* sequence. For a moved block that is exactly the
    approximation under test; for a segment matched against an unrelated identical
    passage elsewhere it is a hazard the matcher cannot see.
    """
    ops = compute_delta(_id_bytes(old_ids), _id_bytes(new_ids), block_size=block_tokens * _W).ops
    segs: list[Segment] = []
    dst = 0
    for op in ops:
        if isinstance(op, Copy):
            src, at, n = op.offset, dst, op.length
            pad = (-src) % _W
            src, at, n = src + pad, at + pad, n - pad
            if at % _W == 0 and n // _W >= min_tokens:
                start = src // _W
                segs.append(Segment(start, start + n // _W, at // _W))
            dst += op.length
        else:
            dst += len(op.data)
    return segs


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


def stitch_segments(
    old_kv: list[tuple[torch.Tensor, torch.Tensor]],
    segments: list[Segment],
    total: int,
    inv_freq: torch.Tensor,
    rotate: bool = True,
) -> ShiftCache:
    """Place every reused segment into a fresh ``total``-long cache, K re-rotated per δ.

    Slots no segment covers are left zeroed; they are the fresh spans, and the caller
    computes them (all of them in one forward — the causal mask makes a fresh token at
    position ``p`` attend to every stitched *and* freshly written slot below ``p``, so
    fresh spans are effectively evaluated in destination order).
    """
    out = []
    for k, v in old_kv:
        nk = k.new_zeros((k.shape[0], k.shape[1], total, k.shape[3]))
        nv = v.new_zeros((v.shape[0], v.shape[1], total, v.shape[3]))
        for seg in segments:
            if seg.length <= 0:
                continue
            src = slice(seg.src_start, seg.src_end)
            dst = slice(seg.dst_start, seg.dst_end)
            nk[:, :, dst] = rerotate_keys(k[:, :, src], seg.delta if rotate else 0, inv_freq)
            nv[:, :, dst] = v[:, :, src]
        out.append((nk, nv))
    return ShiftCache(out)


def fresh_positions(segments: list[Segment], total: int, device) -> torch.Tensor:
    """Destination positions no segment covers — exactly what must be recomputed."""
    covered = torch.zeros(total, dtype=torch.bool, device=device)
    for seg in segments:
        covered[seg.dst_start : seg.dst_end] = True
    return (~covered).nonzero(as_tuple=True)[0]


def stitch(
    old_kv: list[tuple[torch.Tensor, torch.Tensor]],
    span: Span,
    tail: int,
    inv_freq: torch.Tensor,
    rotate: bool = True,
) -> ShiftCache:
    """Single-edit convenience: P verbatim, E' fresh, S re-rotated, ``tail`` slots fresh."""
    return stitch_segments(old_kv, span_segments(span), span.new_len + tail, inv_freq, rotate)


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
def _deviation(model, cache: ShiftCache, ids, positions, total, slots, check_layer):
    """Fresh-vs-cached K deviation at ``check_layer`` for every position in ``slots``.

    Costs ``(check_layer+1)/n_layers`` of a full prefill over the selected tokens —
    charged honestly in the reported effective fraction.
    """
    probe = cache.clone()
    probe.index = positions
    cached = probe.layers[check_layer].keys[:, :, slots].clone()
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
    fresh = probe.layers[check_layer].keys[:, :, slots]
    return (fresh.float() - cached.float()).norm(dim=-1).mean(dim=(0, 1))


@torch.no_grad()
def select(model, policy: Policy, cache: ShiftCache, segments, fresh, all_ids, total):
    """Extra destination positions to recompute, and the extra-cost fraction paid.

    Only segments with something before them are candidates: a leading segment at
    destination 0 is a true prefix, and nothing about its attention changed.
    """
    device = all_ids.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    repairable = [seg for seg in segments if seg.dst_start > 0 and seg.length > 0]
    if policy.kind == "none" or not repairable:
        return empty, 0.0
    if policy.kind == "firstm":
        picked = [
            torch.arange(seg.dst_start, min(seg.dst_start + policy.m, seg.dst_end), device=device)
            for seg in repairable
        ]
        return torch.cat(picked).sort().values, 0.0
    if policy.kind == "blend":
        cand = torch.cat(
            [torch.arange(seg.dst_start, seg.dst_end, device=device) for seg in repairable]
        )
        probe_pos = torch.cat([fresh, cand]).sort().values
        score = _deviation(
            model, cache, all_ids[probe_pos], probe_pos, total, cand, policy.check_layer
        )
        n = max(1, int(round(policy.ratio * cand.numel())))
        return cand[score.topk(n).indices].sort().values, (
            policy.check_layer + 1
        ) / model.config.num_hidden_layers
    raise ValueError(policy.kind)


# --------------------------------------------------------------------------- driver


@torch.no_grad()
def run_segments(
    model,
    old_kv,
    segments: list[Segment],
    all_ids: torch.Tensor,
    policy: Policy,
    max_new_tokens: int = 16,
    forced: list[int] | None = None,
) -> dict:
    """Stitch an arbitrary set of reused segments, recompute the rest, answer."""
    import time

    device = all_ids.device
    total = int(all_ids.shape[0])
    cache = stitch_segments(old_kv, segments, total, inv_freq_of(model), rotate=policy.rerotate)
    fresh = fresh_positions(segments, total, device)
    picked, extra = select(model, policy, cache, segments, fresh, all_ids, total)
    positions = torch.cat([fresh, picked]).sort().values if picked.numel() else fresh

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
        "segments": len(segments),
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
    """Single-edit convenience wrapper over :func:`run_segments`."""
    return run_segments(
        model,
        old_kv,
        span_segments(span),
        torch.cat([new_ids, query_ids]),
        policy,
        max_new_tokens,
        forced,
    )


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
        "segments": 0,
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
