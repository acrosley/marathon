"""Delta-driven reuse on a *hybrid* model: 3/4 of the layers have no KV at all.

``kvshift.py`` answers the dense-model question: when an edit turns

    old   [ P ][ E  ][ S ]        into        new   [ P ][ E' ][ S ]

you keep P's KV, compute E' fresh, and move S's keys by ``d = |E'|-|E|`` with one
extra RoPE rotation. Every layer is full attention, so "reuse" means "reuse KV".

Qwen3.5-4B is not that model. 24 of its 32 layers are Gated DeltaNet: a *recurrent*
token mixer carrying a fixed-size state ``[heads, k_dim, v_dim]`` plus a 4-wide conv
window. There is no per-token KV to re-rotate, and no position to re-rotate it to —
the state is a running summary, and the only thing that "moves S by d" could mean is
"run S's tokens through the recurrence again". So the dense trick applies to 8 layers
and something else has to happen on the other 24. This module is that something else.

Policies, all sharing one machinery (a fresh chunk, then a stale replay chunk):

``stale-state`` (A)
    The naive serving baseline. Attention layers get the full kvshift treatment.
    Linear layers hand the query the recurrent state cached at the END of the old
    context — so the linear half of the model never sees E' at all. Cheapest
    possible; the question is whether the 8 attention layers can carry the edit.

``replay-hidden`` (B)
    Cache, from the old turn, each linear layer's per-token *input* hidden states
    over S (the post-``input_layernorm`` residual stream entering ``linear_attn``),
    plus the state at the end of P. On the edit: run E' fresh through all layers
    (attention layers attend to P's stitched KV; linear layers start from the
    end-of-P state), then for each linear layer roll ONLY its recurrence over S
    from those stale inputs, starting from the fresh post-E' state. Per-token
    inputs are stale, the aggregation is fresh — exactly the staleness class of
    re-rotated KV. Cost is the layer's input projections + the O(d^2) scan, and
    crucially *not* the MLP, which is 63% of a linear layer's weights.

``replay-mix`` (B2)
    Same idea, one cache level deeper: store the old turn's post-conv
    ``(q, k, v, beta, g)`` for S instead of the hidden states. That is the linear
    layer's true analogue of a KV cache, and replay collapses to the bare
    recurrence — no projections at all, ~1.4% of a token-forward instead of ~15%.
    The conv window at S's first 3 tokens still reaches back into the *old* E,
    which ``replay-hidden`` refreshes and this does not.

``replay-first{M}`` (C)
    ``replay-hidden`` with the fresh chunk extended past E' into the first M tokens
    of S, so the tokens closest to the edit are recomputed rather than replayed.

``no-rerotate``
    Control: ``stale-state`` with S's attention keys left at their old angles.

Nothing here changes ``kvshift.py``; the RoPE helpers are imported from it, with
one addition — Qwen3.5 has ``partial_rotary_factor=0.25``, so only the first 64 of
each 256-wide head is rotated and :func:`rerotate_keys_partial` must leave the rest
alone. (Its mRoPE is a no-op for text: all three position rows are equal, so the
interleaved sections carry identical frequencies and it degenerates to plain RoPE.
:func:`check_rope` asserts that on the real model rather than trusting it.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, LinearAttentionLayer

from .kvshift import Span, _ScatterLayer, rerotate_keys

# --------------------------------------------------------------------------- RoPE


def rerotate_keys_partial(k: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """:func:`kvshift.rerotate_keys` on a partial-rotary head: only the rotated dims move.

    ``k`` is ``[..., seq, head_dim]`` with ``head_dim >= 2*len(inv_freq)``. The tail
    dims never carried a position, so moving them would be a bug, not an omission.
    """
    rot = 2 * inv_freq.shape[0]
    if delta == 0 or rot == 0:
        return k
    if rot == k.shape[-1]:
        return rerotate_keys(k, delta, inv_freq)
    head = rerotate_keys(k[..., :rot], delta, inv_freq)
    return torch.cat([head, k[..., rot:]], dim=-1)


def text_position_embeddings(inner, hidden, positions: torch.Tensor):
    """``(cos, sin)`` from the model's own rotary module for a 1-D text position vector."""
    return inner.rotary_emb(hidden, positions[None, None, :].expand(3, 1, -1))


def check_rope(inner, device, n: int = 32) -> float:
    """Max |mRoPE(p+d) - rerotate(mRoPE(p), d)| on the real module. Should be ~0."""
    inv = inner.rotary_emb.inv_freq.detach()
    rot = 2 * inv.shape[0]
    k = torch.randn(1, 4, n, rot, device=device, dtype=torch.float32)

    def rope(p):
        pos = torch.arange(p, p + n, device=device)
        cos, sin = text_position_embeddings(inner, k, pos)
        from .kvshift import rotate_half

        c, s = cos[0].float(), sin[0].float()
        return k * c + rotate_half(k) * s

    return float((rerotate_keys(rope(100), 37, inv) - rope(137)).abs().max())


# ------------------------------------------------------------------ layer taxonomy


def layer_kinds(inner) -> list[str]:
    return list(inner.config.layer_types)


def _params(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


@dataclass(frozen=True)
class CostModel:
    """Per-token weight-FLOP proxies (parameter counts; MACs are 2x either way).

    ``full_token`` is one token through the whole model. ``replay_hidden`` and
    ``replay_mix`` are what one *replayed* S token costs across all linear layers
    under policies B and B2. The recurrence itself is weightless, so its
    ``heads * k_dim * v_dim`` state ops are added explicitly — at 32x128x128 per
    linear layer it is not negligible next to a 21M-parameter projection.
    """

    full_token: int
    replay_hidden: int
    replay_mix: int
    n_linear: int
    n_attn: int

    @classmethod
    def of(cls, inner) -> CostModel:
        kinds = layer_kinds(inner)
        cfg = inner.config
        # ~4 state-sized matmuls per token in the chunked scan (v', decay, outer product)
        scan = 4 * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
        full = hidden = mix = 0
        n_lin = n_att = 0
        for i, layer in enumerate(inner.layers):
            full += _params(layer.mlp)
            if kinds[i] == "linear_attention":
                n_lin += 1
                gdn = layer.linear_attn
                full += _params(gdn) + scan
                proj = (
                    _params(gdn.in_proj_qkv)
                    + _params(gdn.in_proj_b)
                    + _params(gdn.in_proj_a)
                    + _params(gdn.conv1d)
                )
                hidden += proj + scan
                mix += scan
            else:
                n_att += 1
                full += _params(layer.self_attn)
        return cls(full, hidden, mix, n_lin, n_att)

    def frac(self, fresh: int, replayed: int, total: int, mode: str) -> float:
        """Fraction of a full recompute's weight FLOPs, for ``total`` new tokens."""
        per = {"none": 0, "hidden": self.replay_hidden, "mix": self.replay_mix}[mode]
        return (fresh * self.full_token + replayed * per) / (total * self.full_token)


# ------------------------------------------------------------------------- caches


class HybridShiftCache(Cache):
    """Attention layers get ``kvshift``'s scatter buffer; linear layers a real state."""

    def __init__(self, layers: list, kinds: list[str]) -> None:
        self._box: list = [None]
        self.kinds = kinds
        super().__init__(layers=layers)

    @property
    def index(self) -> torch.Tensor | None:
        return self._box[0]

    @index.setter
    def index(self, value: torch.Tensor | None) -> None:
        self._box[0] = value

    @property
    def is_linear(self) -> list[bool]:
        return [k == "linear_attention" for k in self.kinds]

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self.kinds)


def _seed_linear(layer, conv: torch.Tensor, rec: torch.Tensor) -> None:
    layer.lazy_initialization(conv_states=conv, recurrent_states=rec)
    layer.conv_states[0].copy_(conv)
    layer.recurrent_states[0].copy_(rec)
    layer.has_previous_state[0] = True


# --------------------------------------------------------------- old-turn capture


@dataclass
class OldContext:
    """Everything the *previous* turn leaves behind for a delta-driven next turn.

    ``kv`` is the dense-model artefact (attention layers only). ``state_p`` /
    ``state_end`` are the recurrent+conv states at the end of P and of the whole old
    sequence. ``s_hidden`` and ``s_mix`` are the two grades of linear-layer cache;
    both cost the *old* turn nothing extra beyond memory, exactly like KV does.
    """

    kinds: list[str]
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    state_p: dict[int, tuple[torch.Tensor, torch.Tensor]]
    state_end: dict[int, tuple[torch.Tensor, torch.Tensor]]
    s_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    s_mix: dict[int, tuple] = field(default_factory=dict)

    def bytes(self) -> dict[str, float]:
        def nb(t):
            return t.numel() * t.element_size()

        kv = sum(nb(k) + nb(v) for k, v in self.kv.values())
        hid = sum(nb(t) for t in self.s_hidden.values())
        mix = sum(nb(t) for tup in self.s_mix.values() for t in tup)
        return {"kv_mib": kv / 2**20, "s_hidden_mib": hid / 2**20, "s_mix_mib": mix / 2**20}


def _fresh_cache(inner) -> Cache:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(config=inner.config)


@torch.no_grad()
def capture_old(model, old_ids: torch.Tensor, span: Span, want_mix: bool = True) -> OldContext:
    """Prefill the old sequence in two chunks (P, then E+S) and keep what reuse needs."""
    inner = model.model
    kinds = layer_kinds(inner)
    lin_idx = [i for i, k in enumerate(kinds) if k == "linear_attention"]
    grabbed: dict[int, list] = {i: [] for i in lin_idx}

    def hook(idx):
        def fn(_mod, _args, kwargs):
            grabbed[idx].append(kwargs["hidden_states"].detach())

        return fn

    handles = [
        inner.layers[i].linear_attn.register_forward_pre_hook(hook(i), with_kwargs=True)
        for i in lin_idx
    ]
    try:
        cache = _fresh_cache(inner)
        kw = {"past_key_values": cache, "use_cache": True, "logits_to_keep": 1}
        model(input_ids=old_ids[None, : span.p], **kw)
        state_p = {
            i: (cache.layers[i].conv_states[0].clone(), cache.layers[i].recurrent_states[0].clone())
            for i in lin_idx
        }
        for i in lin_idx:
            grabbed[i].clear()
        model(input_ids=old_ids[None, span.p :], **kw)
    finally:
        for h in handles:
            h.remove()

    state_end = {
        i: (cache.layers[i].conv_states[0].clone(), cache.layers[i].recurrent_states[0].clone())
        for i in lin_idx
    }
    kv = {
        i: (cache.layers[i].keys.clone(), cache.layers[i].values.clone())
        for i, k in enumerate(kinds)
        if k == "full_attention"
    }
    out = OldContext(kinds=kinds, kv=kv, state_p=state_p, state_end=state_end)
    for i in lin_idx:
        tail = torch.cat(grabbed[i], dim=1)  # [1, e_old + s, hidden]
        out.s_hidden[i] = tail[:, span.e_old :].clone()
        if want_mix:
            gdn = inner.layers[i].linear_attn
            q, k_, v, beta, g, _ = gdn_prep(gdn, tail, state_p[i][0])
            out.s_mix[i] = tuple(x[:, span.e_old :].clone() for x in (q, k_, v, beta, g))
    del cache
    return out


# ------------------------------------------------------- GatedDeltaNet, taken apart


def gdn_prep(gdn, hidden: torch.Tensor, conv_state: torch.Tensor | None):
    """Everything ``Qwen3_5GatedDeltaNet.forward`` does before the recurrence.

    Mirrors the upstream forward exactly (same conv left-context handling, same
    ``beta``/``g``, same head repeat) but skips ``in_proj_z``, the gated norm and
    ``out_proj`` — none of which affect the recurrent state. Returns the new conv
    state alongside, so a caller can chain chunks.
    """
    from torch.nn.functional import softplus
    from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

    n = hidden.shape[1]
    mixed = gdn.in_proj_qkv(hidden).transpose(1, 2)  # [1, conv_dim, n]
    b = gdn.in_proj_b(hidden)
    a = gdn.in_proj_a(hidden)
    full = mixed if conv_state is None else torch.cat([conv_state, mixed], dim=-1)
    conv = causal_conv1d_fn(
        full, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, activation=gdn.activation
    )[:, :, -n:]
    conv = conv.transpose(1, 2)
    q, k, v = torch.split(conv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    q = q.reshape(1, n, -1, gdn.head_k_dim)
    k = k.reshape(1, n, -1, gdn.head_k_dim)
    v = v.reshape(1, n, -1, gdn.head_v_dim)
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * softplus(a.float() + gdn.dt_bias)
    rep = gdn.num_v_heads // gdn.num_k_heads
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    return q, k, v, beta, g, full[..., -gdn.conv_kernel_size :].clone()


def gdn_state(gdn, q, k, v, beta, g, initial_state: torch.Tensor) -> torch.Tensor:
    """Roll the recurrence and return only the final state — no output projection."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    _, last = torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    return last


# ------------------------------------------------------------------------ policies


@dataclass
class HybridPolicy:
    """``linear`` in {stale, hidden, mix}; ``first_m`` extends the fresh chunk into S."""

    linear: str = "hidden"
    first_m: int = 0
    rerotate: bool = True
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        if not self.rerotate:
            return "no-rerotate"
        base = {"stale": "stale-state", "hidden": "replay-hidden", "mix": "replay-mix"}[self.linear]
        return f"{base}+first{self.first_m}" if self.first_m else base

    @property
    def mode(self) -> str:
        return {"stale": "none", "hidden": "hidden", "mix": "mix"}[self.linear]


# --------------------------------------------------------------------- the forward


def _causal_mask(positions: torch.Tensor, total: int, dtype: torch.dtype) -> torch.Tensor:
    cols = torch.arange(total, device=positions.device)
    allowed = cols[None, :] <= positions[:, None]
    mask = torch.zeros(allowed.shape, dtype=dtype, device=positions.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


@torch.no_grad()
def _layer_loop(model, cache, ids, positions, total, replay, keep: int = 1, scatter: bool = False):
    """One chunk through every layer, with an optional per-linear-layer stale replay.

    ``replay`` maps a linear layer index to the arguments of its S rollout. It runs
    *inside* the layer, right after the fresh chunk has advanced that layer's state,
    which is what makes the stitched state ordered P -> E' -> S even though the
    layers are visited top to bottom. ``scatter`` writes the chunk's KV into its
    destination slots of the pre-stitched buffer instead of appending.
    """
    inner = model.model
    kinds = cache.kinds
    h = inner.embed_tokens(ids[None])
    cos, sin = text_position_embeddings(inner, h, positions)
    mask = _causal_mask(positions, total, h.dtype)
    cache.index = positions if scatter else None
    for i, layer in enumerate(inner.layers):
        residual = h
        x = layer.input_layernorm(h)
        if kinds[i] == "linear_attention":
            out = layer.linear_attn(hidden_states=x, cache_params=cache)
            if replay is not None and i in replay:
                lyr = cache.layers[i]
                gdn = layer.linear_attn
                state = lyr.recurrent_states[0]
                kind, payload = replay[i]
                if kind == "hidden":
                    q, k, v, beta, g, conv = gdn_prep(gdn, payload, lyr.conv_states[0])
                    lyr.conv_states[0].copy_(conv)
                else:
                    q, k, v, beta, g = payload
                lyr.recurrent_states[0].copy_(gdn_state(gdn, q, k, v, beta, g, state))
        else:
            out, _ = layer.self_attn(
                hidden_states=x,
                attention_mask=mask,
                position_ids=positions[None],
                past_key_values=cache,
                position_embeddings=(cos, sin),
            )
        h = residual + out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    cache.index = None
    h = inner.norm(h[:, -keep:])
    return model.lm_head(h)[0]


def _build_cache(model, old: OldContext, span: Span, tail: int, rotate: bool, stale_end: bool):
    """Stitched attention KV (P verbatim, S re-rotated by delta) + seeded linear states."""
    inner = model.model
    inv = inner.rotary_emb.inv_freq.detach()
    total = span.new_len + tail
    layers: list = []
    for i, kind in enumerate(old.kinds):
        if kind == "full_attention":
            k, v = old.kv[i]
            nk = k.new_zeros((k.shape[0], k.shape[1], total, k.shape[3]))
            nv = v.new_zeros((v.shape[0], v.shape[1], total, v.shape[3]))
            if span.p:
                nk[:, :, : span.p] = k[:, :, : span.p]
                nv[:, :, : span.p] = v[:, :, : span.p]
            if span.s:
                src = slice(span.p + span.e_old, span.p + span.e_old + span.s)
                dst = slice(span.p + span.e_new, span.new_len)
                d = span.delta if rotate else 0
                nk[:, :, dst] = rerotate_keys_partial(k[:, :, src], d, inv)
                nv[:, :, dst] = v[:, :, src]
            layers.append(_ScatterLayer(nk, nv, [None]))
        else:
            layers.append(LinearAttentionLayer())
    cache = HybridShiftCache(layers, old.kinds)
    for lyr in layers:
        if isinstance(lyr, _ScatterLayer):
            lyr.box = cache._box
    src = old.state_end if stale_end else old.state_p
    for i, kind in enumerate(old.kinds):
        if kind == "linear_attention":
            conv, rec = src[i]
            _seed_linear(layers[i], conv.clone(), rec.clone())
    return cache, total


@torch.no_grad()
def run_hybrid(
    model,
    old: OldContext,
    span: Span,
    new_ids: torch.Tensor,
    query_ids: torch.Tensor,
    policy: HybridPolicy,
    max_new_tokens: int = 12,
    forced: list[int] | None = None,
) -> dict:
    """Answer ``query_ids`` from a delta-reused hybrid cache. Mirrors ``kvshift.run_policy``."""
    device = new_ids.device
    cost = CostModel.of(model.model)
    stale = policy.linear == "stale"
    cache, total = _build_cache(model, old, span, 0, policy.rerotate, stale_end=stale)

    m = min(policy.first_m, span.s) if policy.first_m else 0
    fresh_lo, fresh_hi = span.p, span.p + span.e_new + m
    fresh_pos = torch.arange(fresh_lo, fresh_hi, device=device)
    replayed = span.s - m

    replay = None
    if not stale and replayed > 0:
        replay = {}
        for i, kind in enumerate(old.kinds):
            if kind != "linear_attention":
                continue
            if policy.linear == "hidden":
                replay[i] = ("hidden", old.s_hidden[i][:, m:])
            else:
                replay[i] = ("mix", tuple(x[:, m:] for x in old.s_mix[i]))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    _layer_loop(model, cache, new_ids[fresh_lo:fresh_hi], fresh_pos, total, replay, scatter=True)
    seen = total + int(query_ids.shape[0])
    q_pos = torch.arange(total, seen, device=device)
    logits = _layer_loop(model, cache, query_ids, q_pos, seen, None)[-1]
    sync()
    prefill_s = time.perf_counter() - t0

    seq = logits[None]
    if forced and len(forced) > 1:
        seq = torch.cat([logits[None], _forced(model, cache, forced[:-1], seen)], 0)
    toks = _greedy(model, cache, logits, max_new_tokens, seen)
    sync()

    n_total = span.new_len + query_ids.shape[0]
    fresh_tokens = int(fresh_pos.numel()) + int(query_ids.shape[0])
    return {
        "policy": policy.label(),
        "fresh_tokens": fresh_tokens,
        "replayed_tokens": replayed if not stale else 0,
        "fresh_frac": fresh_tokens / n_total,
        "flop_frac": cost.frac(fresh_tokens, replayed if not stale else 0, n_total, policy.mode),
        "prefill_s": prefill_s,
        "wall_s": time.perf_counter() - t0,
        "tokens": toks,
        "logits": logits,
        "logits_seq": seq,
    }


@torch.no_grad()
def _forced(model, cache, ref_tokens: list[int], seen: int) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor(ref_tokens, device=device)
    pos = torch.arange(seen, seen + len(ref_tokens), device=device)
    probe = _clone_cache(cache)
    return _layer_loop(model, probe, ids, pos, seen + len(ref_tokens), None, keep=len(ref_tokens))


@torch.no_grad()
def _greedy(model, cache, logits, n: int, seen: int) -> list[int]:
    device = logits.device
    toks: list[int] = []
    for i in range(n):
        nxt = int(logits.argmax())
        toks.append(nxt)
        if i == n - 1:
            break
        pos = torch.tensor([seen + i], device=device)
        logits = _layer_loop(
            model, cache, torch.tensor([nxt], device=device), pos, seen + i + 1, None
        )[-1]
    return toks


def _clone_cache(cache: HybridShiftCache) -> HybridShiftCache:
    layers: list = []
    for lyr in cache.layers:
        if isinstance(lyr, _ScatterLayer):
            layers.append(_ScatterLayer(lyr.keys.clone(), lyr.values.clone(), [None]))
        else:
            new = LinearAttentionLayer()
            _seed_linear(new, lyr.conv_states[0].clone(), lyr.recurrent_states[0].clone())
            layers.append(new)
    out = HybridShiftCache(layers, cache.kinds)
    for lyr in layers:
        if isinstance(lyr, _ScatterLayer):
            lyr.box = out._box
    return out


# ------------------------------------------------------------------- the reference


@torch.no_grad()
def run_full_hybrid(model, new_ids, query_ids, max_new_tokens: int = 12) -> dict:
    """Full recompute of the whole new sequence — ground truth for everything above."""
    ids = torch.cat([new_ids, query_ids])
    device = ids.device
    n = int(ids.shape[0])

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    cache = _fresh_cache(model.model)
    logits = model(
        input_ids=ids[None], past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits[0, -1]
    sync()
    prefill_s = time.perf_counter() - t0
    toks: list[int] = []
    seq = [logits]
    for i in range(max_new_tokens):
        nxt = int(logits.argmax())
        toks.append(nxt)
        if i == max_new_tokens - 1:
            break
        out = model(
            input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache, use_cache=True
        )
        logits = out.logits[0, -1]
        seq.append(logits)
    sync()
    return {
        "policy": "full-recompute",
        "fresh_tokens": n,
        "replayed_tokens": 0,
        "fresh_frac": 1.0,
        "flop_frac": 1.0,
        "prefill_s": prefill_s,
        "wall_s": time.perf_counter() - t0,
        "tokens": toks,
        "logits": logits,
        "logits_seq": torch.stack(seq),
    }
