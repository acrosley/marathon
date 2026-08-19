"""Phase 1 probe: prefill cost per turn on self-hosted vLLM, with and without
non-prefix KV reuse (LMCache CacheBlend).

Same session shape as ``live_probe`` (append-only turns, optional in-place
edit of turn 0), but served locally so we own the KV cache. Three modes:

    none    no caching at all: every turn re-prefills the whole history
    prefix  vLLM native prefix caching (the Phase 0 provider behaviour)
    blend   LMCache CacheBlend: each message is a chunk delimited by a special
            separator; chunk KV is reused regardless of position, with a
            small fraction of tokens recomputed to re-stitch attention
    shift   Marathon position-shifted KV reuse (``vllm_shift_connector``): prefix
            caching stays on, and on the edit turn the unchanged suffix ``S`` is
            copied into its new block slots with its keys re-rotated by the token
            shift, so vLLM prefills only the edited message and the new turn

Wall time of a ``max_tokens=1`` generate is the prefill cost (offline engine,
no network). Runs inside WSL2 in ``~/marathon-venv`` (see
``scripts/phase1_setup.sh``); not run in CI, no tests. Usage:

    python -m marathon.local_probe --mode prefix --turns 16 --edit-at 13
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

from .session import Session

_FILLER = (
    "This is deterministic filler content used to grow the context in a "
    "byte-stable, append-only fashion so that KV reuse can be measured across turns. "
) * 20  # ~300 tokens/turn, above LMCache's 256-token blend minimum

_SYSTEM = "You are a latency probe. Reply to every message with the single word: ok"
_SEP = " # # "  # LMCache blend separator; also used as message delimiter in every mode

# Parity probe: a unique fact planted mid-history, asked about on the last turn.
# Answering it requires actually reading turn 3's KV, so a wrong/blank answer
# means the reused KV lost information the full recompute keeps.
_PARITY_FACT = "The access code is 7391-KAPPA."
_PARITY_AT = 3
_PARITY_QUESTION = "What is the access code? Answer with only the code."

# Prepended by --edit-grow to make the edit *grow* the history rather than shift it
# by a handful of tokens: a bigger delta is the harder case for re-rotation.
_GROW = "Amended note: this span was revised later in the session. "


def _configure(mode: str, recompute_ratio: float) -> None:
    if mode != "blend":
        return
    os.environ.update(
        LMCACHE_CHUNK_SIZE="256",
        LMCACHE_LOCAL_CPU="True",
        LMCACHE_MAX_LOCAL_CPU_SIZE="20",
        LMCACHE_ENABLE_BLENDING="True",
        LMCACHE_BLEND_SPECIAL_STR=_SEP,
        LMCACHE_USE_LAYERWISE="True",
        LMCACHE_BLEND_CHECK_LAYERS="1",
        LMCACHE_BLEND_RECOMPUTE_RATIOS=str(recompute_ratio),
    )


def _reuse_plan(old_chunks: list[list[int]], new_chunks: list[list[int]], block_size: int):
    """``(phase1_len, dst_end, delta)`` for a single edited chunk, else ``None``.

    The prompt is a list of chunks (system, one per message, trailing tail). An
    in-place edit changes exactly one chunk and shifts every following chunk by
    ``delta``; the run of chunks that survives unchanged is the reusable ``S``.
    ``phase1_len`` rounds the start of ``S`` up to a block boundary — vLLM only
    reports prefix hits in whole blocks, so the first partial block of ``S`` is
    prefilled along with ``E'`` instead of being copied.
    """
    a = 0
    while a < len(old_chunks) and a < len(new_chunks) and old_chunks[a] == new_chunks[a]:
        a += 1
    if a == 0 or a + 1 >= min(len(old_chunks), len(new_chunks)):
        return None
    delta = len(new_chunks[a]) - len(old_chunks[a])
    k = 0
    while (
        a + 1 + k < min(len(old_chunks), len(new_chunks))
        and old_chunks[a + 1 + k] == new_chunks[a + 1 + k]
    ):
        k += 1
    if k == 0:
        return None
    src_start = sum(len(c) for c in old_chunks[: a + 1])
    src_end = src_start + sum(len(c) for c in old_chunks[a + 1 : a + 1 + k])
    dst_start, dst_end = src_start + delta, src_end + delta
    phase1_len = -(-dst_start // block_size) * block_size
    return (phase1_len, dst_end, delta) if dst_end - phase1_len >= block_size else None


@contextlib.contextmanager
def _engine(mode: str, model: str, max_model_len: int, blend_prefix: bool = False):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kwargs: dict = dict(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.80 if mode == "shift" else 0.85,
        enable_prefix_caching=(mode in ("prefix", "shift") or (mode == "blend" and blend_prefix)),
        enforce_eager=(mode == "blend"),  # matches LMCache blend example
        disable_log_stats=False,
    )
    if mode == "shift":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="MarathonShiftConnector",
            kv_connector_module_path="marathon.vllm_shift_connector",
            kv_role="kv_both",
        )
    if mode == "blend":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        )
    llm = LLM(**kwargs)
    try:
        yield llm
    finally:
        if mode == "blend":
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder

            LMCacheEngineBuilder.destroy(ENGINE_NAME)


def _prefix_hits(llm) -> tuple[int, int] | None:
    """(hit_tokens, query_tokens) cumulative from vLLM's own prefix-cache counters."""
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None
    vals = {m.name: getattr(m, "value", None) for m in metrics}
    h, q = vals.get("vllm:prefix_cache_hits"), vals.get("vllm:prefix_cache_queries")
    return (int(h), int(q)) if h is not None and q is not None else None


def probe(
    mode: str,
    model: str,
    turns: int,
    edit_at: int | None,
    max_model_len: int,
    recompute_ratio: float = 0.15,
    blend_prefix: bool = False,
    parity_tokens: int = 0,
    edit_turn: int = 0,
    edit_grow: int = 0,
    repair_first: int = 0,
) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    _configure(mode, recompute_ratio)
    tok = AutoTokenizer.from_pretrained(model)
    # Mirror LMCache's SegmentTokenDatabase exactly: it strips the config value
    # and drops the first token (``encode(blend_special_str)[1:]``). Deriving the
    # delimiter any other way emits a token LMCache also splits on, and two
    # adjacent split points produce a zero-length chunk -> ZeroDivisionError in
    # its allocator.
    sep = tok.encode(_SEP.strip())[1:]
    sys_ids = tok.encode(_SYSTEM, add_special_tokens=False)
    tail = tok.encode("\nassistant: ", add_special_tokens=False)
    # Qwen3 is a thinking model; on the parity turn prefill an empty (closed)
    # think block so the few generated tokens are the answer, not reasoning.
    tail_answer = tok.encode("\nassistant: <think>\n\n</think>\n\n", add_special_tokens=False)
    sampling = SamplingParams(temperature=0, max_tokens=1)
    parity_sampling = SamplingParams(temperature=0, max_tokens=max(parity_tokens, 1))
    session = Session()
    rows: list[dict] = []

    def _sp(base: SamplingParams, **kv) -> SamplingParams:
        """Copy of ``base`` carrying a reuse plan for the shift connector."""
        if mode != "shift":
            return base
        p = base.clone()
        p.extra_args = {"kv_transfer_params": kv}
        return p

    with _engine(mode, model, max_model_len, blend_prefix) as llm:
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        # warm the engine (CUDA graphs / kernels) so turn 0 isn't inflated
        llm.generate({"prompt_token_ids": sys_ids}, _sp(sampling, save=False))
        prev = _prefix_hits(llm) or (0, 0)
        prev_chunks: list[list[int]] | None = None

        for t in range(turns):
            if t == edit_at:
                i = 2 * edit_turn
                grow = _GROW * max(edit_grow // len(tok.encode(_GROW, add_special_tokens=False)), 0)
                session.edit(i, "[EDITED] " + grow + session.messages[i]["content"])
            last = t == turns - 1
            ask = parity_tokens > 0 and last
            fact = _PARITY_FACT + " " if parity_tokens > 0 and t == _PARITY_AT else ""
            request = _PARITY_QUESTION if ask else "Reply 'ok'."
            state = session.turn("user", f"Turn {t}. {fact}{_FILLER} {request}")
            history = Session.decode(state)

            # prompt = system, then each message as its own separator-delimited chunk
            chunks = [list(sys_ids)]
            for h in history:
                chunks.append(
                    sep + tok.encode(f"{h['role']}: {h['content']}", add_special_tokens=False)
                )
            chunks.append(sep + (tail_answer if ask else tail))
            ids = [i for c in chunks for i in c]

            base = parity_sampling if ask else sampling
            # save the KV of everything computed while history is still append-only;
            # the store is what the edit turn re-rotates out of.
            save = mode == "shift" and (edit_at is None or t < edit_at)
            plan = (
                _reuse_plan(prev_chunks, chunks, block_size)
                if mode == "shift" and t == edit_at and prev_chunks
                else None
            )
            prev_chunks = chunks

            start = time.perf_counter()
            if plan is not None:
                phase1_len, dst_end, delta = plan
                # phase 1: prefill P + E' at native speed so it lands in the prefix cache
                llm.generate({"prompt_token_ids": ids[:phase1_len]}, _sp(sampling, save=False))
                # phase 2 (repair): prefill the first M tokens of S natively, so they
                # attend to E' instead of carrying attention to the replaced E. vLLM's
                # connector API can only express matched tokens as a prefix, so the
                # repaired head has to be a separate, block-aligned request.
                load_from = phase1_len
                if repair_first > 0:
                    repair = -(-repair_first // block_size) * block_size
                    load_from = min(phase1_len + repair, dst_end - block_size)
                    load_from -= load_from % block_size
                    llm.generate({"prompt_token_ids": ids[:load_from]}, _sp(sampling, save=False))
                # final phase: prefix-hit everything computed so far, connector supplies
                # the rest of S re-rotated, vLLM prefills the new turn and the query
                out = llm.generate(
                    {"prompt_token_ids": ids},
                    _sp(
                        base,
                        save=False,
                        load={"dst_start": load_from, "dst_end": dst_end, "delta": delta},
                    ),
                )
            else:
                out = llm.generate({"prompt_token_ids": ids}, _sp(base, save=save))
            prefill_s = time.perf_counter() - start

            cur = _prefix_hits(llm)
            hit = (cur[0] - prev[0]) if cur else None
            prev = cur or prev
            rows.append(
                {
                    "turn": t,
                    "prefill_s": round(prefill_s, 4),
                    "prompt_tokens": len(ids),
                    "prefix_hit_tokens": hit,
                    "wire_bytes": len(session.last_payload.wire_bytes()),
                    "state_bytes": len(state),
                    "text": out[0].outputs[0].text,
                }
            )
            session.turn("assistant", "ok")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marathon.local_probe", description=__doc__)
    parser.add_argument("--mode", choices=["none", "prefix", "blend", "shift"], default="prefix")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--edit-at", type=int, default=None, help="mutate a turn at this turn")
    parser.add_argument(
        "--edit-turn",
        type=int,
        default=0,
        help="which turn's user message --edit-at mutates (default 0)",
    )
    parser.add_argument(
        "--edit-grow",
        type=int,
        default=0,
        help="make the edit add roughly this many tokens (default: a ~4-token shift)",
    )
    parser.add_argument(
        "--repair-first",
        type=int,
        default=0,
        help="shift mode: natively recompute the first M tokens of the reused span",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--recompute-ratio",
        type=float,
        default=0.15,
        help="LMCACHE_BLEND_RECOMPUTE_RATIOS (blend mode)",
    )
    parser.add_argument(
        "--blend-prefix", action="store_true", help="blend mode with vLLM prefix caching on as well"
    )
    parser.add_argument(
        "--parity-tokens",
        type=int,
        default=0,
        help="if >0, ask the planted-fact question on the last turn, N tokens",
    )
    parser.add_argument("--json", default=None, help="write rows to this path")
    args = parser.parse_args(argv)
    rows = probe(
        args.mode,
        args.model,
        args.turns,
        args.edit_at,
        args.max_model_len,
        args.recompute_ratio,
        args.blend_prefix,
        args.parity_tokens,
        args.edit_turn,
        args.edit_grow,
        args.repair_first,
    )
    print(
        f"mode={args.mode} model={args.model} edit_at={args.edit_at} "
        f"ratio={args.recompute_ratio} blend_prefix={args.blend_prefix} "
        f"edit_turn={args.edit_turn} edit_grow={args.edit_grow} repair_first={args.repair_first}"
    )
    cols = ["turn", "prefill_s", "prompt_tokens", "prefix_hit_tokens", "wire_bytes", "state_bytes"]
    print(" ".join(f"{c:>17}" for c in cols), " text")
    for r in rows:
        print(" ".join(f"{str(r[c]):>17}" for c in cols), f" {r['text']!r}")
    if args.json:
        report = {
            "mode": args.mode,
            "model": args.model,
            "edit_at": args.edit_at,
            "recompute_ratio": args.recompute_ratio,
            "blend_prefix": args.blend_prefix,
            "rows": rows,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
