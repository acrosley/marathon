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
            caching stays on, and on the edit turn every unchanged run of history is
            copied into its new block slots with its keys re-rotated by *its own*
            token shift, so vLLM prefills only the edited messages and the new turn.
            ``--edit-count k`` edits k different messages in one turn and ``--move``
            swaps two earlier messages (which gives some segments a negative delta).
            ``--sessions N`` runs N independent sessions interleaved turn by turn,
            each with its own planted fact and its own store keyed by session id; a
            store that leaked across sessions would answer with the wrong code.

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

from .reuse_plan import phases as _phases
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
_PARITY_CODES = ("7391-KAPPA", "5820-OMEGA", "1146-SIGMA", "9032-DELTA")
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


def _line_tokens(tok, sep: list[int]):
    """Tokens one canonical history line contributes to the prompt (probe's layout)."""

    def f(line: bytes) -> list[int]:
        h = json.loads(line)
        return sep + tok.encode(f"{h['role']}: {h['content']}", add_special_tokens=False)

    return f


def _mutate(session, edit_turn: int, count: int, grow: str, move: bool, upto: int) -> None:
    """Apply the edit-turn mutation: ``count`` in-place edits and optionally one swap."""
    available = list(range(edit_turn, upto))
    step = max(len(available) // max(count, 1), 1)
    for t in available[::step][:count]:
        i = 2 * t
        session.edit(i, "[EDITED] " + grow + session.messages[i]["content"])
    if move and upto >= 4:
        a, b = 2, 2 * (upto - 2)
        ca, cb = session.messages[a]["content"], session.messages[b]["content"]
        session.edit(a, cb)
        session.edit(b, ca)


@contextlib.contextmanager
def _engine(
    mode: str,
    model: str,
    max_model_len: int,
    blend_prefix: bool = False,
    store_tokens: int = 16384,
    gpu_util: float = 0.0,
):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kwargs: dict = dict(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util or (0.80 if mode == "shift" else 0.85),
        enable_prefix_caching=(mode in ("prefix", "shift") or (mode == "blend" and blend_prefix)),
        enforce_eager=(mode == "blend"),  # matches LMCache blend example
        disable_log_stats=False,
    )
    if mode == "shift":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="MarathonShiftConnector",
            kv_connector_module_path="marathon.vllm_shift_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"store_tokens": store_tokens},
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
    edit_count: int = 1,
    move: bool = False,
    reuse_moved: bool = False,
    n_sessions: int = 1,
    store_tokens: int = 16384,
    gpu_util: float = 0.0,
) -> list[dict]:
    import dataclasses

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from . import reuse_plan

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
    # One independent conversation per session id. With --sessions 2 the turns are
    # interleaved (s0 turn 0, s1 turn 0, s0 turn 1, ...), both sessions plant a
    # *different* fact and both are edited, so a store that leaked across sessions
    # would answer with the other session's code.
    convos = [
        {
            "name": f"s{i}",
            "session": Session(),
            "prev": None,
            "code": _PARITY_CODES[i % len(_PARITY_CODES)],
        }
        for i in range(max(n_sessions, 1))
    ]
    rows: list[dict] = []

    def _sp(base: SamplingParams, name: str, **kv) -> SamplingParams:
        """Copy of ``base`` carrying this session's reuse plan for the connector."""
        if mode != "shift":
            return base
        p = base.clone()
        p.extra_args = {"kv_transfer_params": {"session": name, **kv}}
        return p

    with _engine(mode, model, max_model_len, blend_prefix, store_tokens, gpu_util) as llm:
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        # warm the engine (CUDA graphs / kernels) so turn 0 isn't inflated
        llm.generate({"prompt_token_ids": sys_ids}, _sp(sampling, "warmup", save=False))
        prev = _prefix_hits(llm) or (0, 0)
        line_tokens = _line_tokens(tok, sep)

        for t in range(turns):
            for convo in convos:
                name, session = convo["name"], convo["session"]
                prev_state = convo["prev"]
                if t == edit_at:
                    per = len(tok.encode(_GROW, add_special_tokens=False))
                    grow = _GROW * max(edit_grow // per, 0)
                    _mutate(session, edit_turn, edit_count, grow, move, t)
                last = t == turns - 1
                ask = parity_tokens > 0 and last
                plant = parity_tokens > 0 and t == _PARITY_AT
                fact = f"The access code is {convo['code']}. " if plant else ""
                request = _PARITY_QUESTION if ask else "Reply 'ok'."
                # The session tag only appears with --sessions >1, so a single-session
                # run stays byte-identical to every earlier measurement in findings.md.
                tag = f"Session {name}. " if len(convos) > 1 else ""
                state = session.turn("user", f"{tag}Turn {t}. {fact}{_FILLER} {request}")
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
                phases: list[tuple[int, dict | None]] = []
                reuse = None
                if mode == "shift" and t == edit_at and prev_state is not None:
                    reuse = reuse_plan.plan(
                        prev_state, state, line_tokens, head_tokens=len(sys_ids)
                    )
                    if repair_first > 0:
                        reuse = dataclasses.replace(reuse, repair_first=repair_first)
                    loads = reuse.to_kv_transfer_params(reuse_moved=reuse_moved)
                    phases = _phases(loads, block_size, len(ids))
                    print(
                        f"[shift] {name} turn {t}: policy={reuse.policy} "
                        f"segments={len(reuse.segments)} "
                        f"deltas={[sg.delta for sg in reuse.segments]} "
                        f"moved={[i for i, m in enumerate(reuse.moved) if m]} "
                        f"reused={sum(sg.length for sg in reuse.segments)}/{len(ids)} "
                        f"requests={len(phases) or 1} ({reuse.reason})",
                        flush=True,
                    )
                convo["prev"] = state

                start = time.perf_counter()
                if phases:
                    # every phase but the last is a max_tokens=1 warm-up whose only job is to
                    # leave its blocks in vLLM's prefix cache for the phase after it
                    for length, load in phases[:-1]:
                        llm.generate(
                            {"prompt_token_ids": ids[:length]},
                            _sp(sampling, name, save=False, **({"load": load} if load else {})),
                        )
                    _, load = phases[-1]
                    out = llm.generate(
                        {"prompt_token_ids": ids}, _sp(base, name, save=False, load=load)
                    )
                else:
                    out = llm.generate({"prompt_token_ids": ids}, _sp(base, name, save=save))
                prefill_s = time.perf_counter() - start

                cur = _prefix_hits(llm)
                hit = (cur[0] - prev[0]) if cur else None
                prev = cur or prev
                rows.append(
                    {
                        "turn": t,
                        "session": name,
                        "prefill_s": round(prefill_s, 4),
                        "prompt_tokens": len(ids),
                        "prefix_hit_tokens": hit,
                        "wire_bytes": len(session.last_payload.wire_bytes()),
                        "state_bytes": len(state),
                        "requests": max(len(phases), 1),
                        "segments": len(reuse.segments) if reuse else 0,
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
        "--edit-count",
        type=int,
        default=1,
        help="how many different messages --edit-at rewrites in the same turn",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="also swap two earlier messages, so some segments get a negative delta",
    )
    parser.add_argument(
        "--reuse-moved",
        action="store_true",
        help="also transplant relocated blocks (measured unsafe; off by default)",
    )
    parser.add_argument(
        "--repair-first",
        type=int,
        default=0,
        help="shift mode: natively recompute the first M tokens of the reused span",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="run N independent sessions interleaved turn by turn, each with its own "
        "planted fact and its own store (proves session isolation in shift mode)",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--store-tokens",
        type=int,
        default=16384,
        help="shift mode: size of the connector's flat KV store, in tokens",
    )
    parser.add_argument(
        "--gpu-util", type=float, default=0.0, help="override gpu_memory_utilization"
    )
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
        args.edit_count,
        args.move,
        args.reuse_moved,
        args.sessions,
        args.store_tokens,
        args.gpu_util,
    )
    print(
        f"mode={args.mode} model={args.model} edit_at={args.edit_at} "
        f"ratio={args.recompute_ratio} blend_prefix={args.blend_prefix} "
        f"edit_turn={args.edit_turn} edit_grow={args.edit_grow} "
        f"repair_first={args.repair_first} edit_count={args.edit_count} move={args.move} "
        f"reuse_moved={args.reuse_moved} sessions={args.sessions}"
    )
    if args.mode == "shift":
        from .vllm_shift_connector import stats

        # vLLM runs the engine core (and with it both connector instances) in a spawned
        # process, so these counters are only populated when the engine happens to be
        # in-process; the connector also logs the same dict, which the run log keeps.
        print("store stats:", json.dumps(stats(), sort_keys=True), "(see log for the engine's own)")
    cols = [
        "turn",
        "session",
        "prefill_s",
        "prompt_tokens",
        "prefix_hit_tokens",
        "requests",
        "segments",
        "state_bytes",
    ]
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
