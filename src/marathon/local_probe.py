"""Phase 1 probe: prefill cost per turn on self-hosted vLLM, with and without
non-prefix KV reuse (LMCache CacheBlend).

Same session shape as ``live_probe`` (append-only turns, optional in-place
edit of turn 0), but served locally so we own the KV cache. Three modes:

    none    no caching at all: every turn re-prefills the whole history
    prefix  vLLM native prefix caching (the Phase 0 provider behaviour)
    blend   LMCache CacheBlend: each message is a chunk delimited by a special
            separator; chunk KV is reused regardless of position, with a
            small fraction of tokens recomputed to re-stitch attention

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


def _configure(mode: str) -> None:
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
        LMCACHE_BLEND_RECOMPUTE_RATIOS="0.15",
    )


@contextlib.contextmanager
def _engine(mode: str, model: str, max_model_len: int):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kwargs: dict = dict(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=(mode == "prefix"),
        enforce_eager=(mode == "blend"),  # matches LMCache blend example
        disable_log_stats=False,
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
) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    _configure(mode)
    tok = AutoTokenizer.from_pretrained(model)
    # Mirror LMCache's SegmentTokenDatabase exactly: it strips the config value
    # and drops the first token (``encode(blend_special_str)[1:]``). Deriving the
    # delimiter any other way emits a token LMCache also splits on, and two
    # adjacent split points produce a zero-length chunk -> ZeroDivisionError in
    # its allocator.
    sep = tok.encode(_SEP.strip())[1:]
    sys_ids = tok.encode(_SYSTEM, add_special_tokens=False)
    tail = tok.encode("\nassistant: ", add_special_tokens=False)
    sampling = SamplingParams(temperature=0, max_tokens=1)
    session = Session()
    rows: list[dict] = []

    with _engine(mode, model, max_model_len) as llm:
        # warm the engine (CUDA graphs / kernels) so turn 0 isn't inflated
        llm.generate({"prompt_token_ids": sys_ids}, sampling)
        prev = _prefix_hits(llm) or (0, 0)

        for t in range(turns):
            if t == edit_at:
                session.edit(0, "[EDITED] " + session.messages[0]["content"])
            state = session.turn("user", f"Turn {t}. {_FILLER} Reply 'ok'.")
            history = Session.decode(state)

            # prompt = system, then each message as its own separator-delimited chunk
            ids = list(sys_ids)
            for h in history:
                ids += sep + tok.encode(f"{h['role']}: {h['content']}", add_special_tokens=False)
            ids += sep + tail

            start = time.perf_counter()
            out = llm.generate({"prompt_token_ids": ids}, sampling)
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
    parser.add_argument("--mode", choices=["none", "prefix", "blend"], default="prefix")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--edit-at", type=int, default=None, help="mutate turn 0 at this turn")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--json", default=None, help="write rows to this path")
    args = parser.parse_args(argv)
    rows = probe(args.mode, args.model, args.turns, args.edit_at, args.max_model_len)
    print(f"mode={args.mode} model={args.model} edit_at={args.edit_at}")
    cols = ["turn", "prefill_s", "prompt_tokens", "prefix_hit_tokens", "wire_bytes", "state_bytes"]
    print(" ".join(f"{c:>17}" for c in cols), " text")
    for r in rows:
        print(" ".join(f"{str(r[c]):>17}" for c in cols), f" {r['text']!r}")
    if args.json:
        report = {"mode": args.mode, "model": args.model, "edit_at": args.edit_at, "rows": rows}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
