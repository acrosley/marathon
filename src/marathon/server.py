"""End-to-end Marathon server: turn payload in, generated reply out.

This is the whole pipeline in one place. A client sends the v0 turn payload
``{baseline_hash, target_hash, delta, new_input}`` for a session id; the server

1. resolves the baseline from its content-addressed store and applies the delta,
   **verifying** ``sha256(result) == target_hash`` before trusting a single byte of it
   (:func:`marathon.protocol.resolve_turn` — reconstruction is proven, never assumed);
2. decodes the verified state into messages and renders them through the model's own
   chat template, one message at a time, so the reuse plan's token coordinates are the
   serving layer's (:class:`ChatTokenizer`);
3. plans KV reuse against the session's *previous* verified state
   (:func:`marathon.reuse_plan.plan`) and turns the reused segments into the k+1
   request phases vLLM's connector API needs (:func:`marathon.reuse_plan.phases`);
4. drives the vLLM offline engine with the shift connector, keyed by session id, and
5. returns the generated text plus per-turn metrics.

Sessions are isolated end to end: the previous state, the piece cache and the
connector's KV store are all keyed by session id, so a plan or a KV read can never
cross sessions.

Saving policy. The connector's store is a flat position-indexed buffer, so it is only
meaningful in *one* set of coordinates at a time. An append-only turn extends it. An
edit turn moves everything after the edit, which would leave the store describing a
layout that no longer exists — so the edit turn's final request saves with ``"full"``,
re-gathering the whole prompt out of the paged KV cache (loaded span, prefix-cache hits
and freshly computed tokens are all resident there by then) at its *new* positions. The
session is append-only again afterwards, which is what makes a session's second, third
and Nth edit cost the same as its first.

Two front doors, both thin: :class:`MarathonServer` is the Python API
(``turn(session_id, payload) -> dict``), and :func:`serve` wraps it in a stdlib
:mod:`http.server` JSON endpoint at ``POST /v1/turn``. stdlib rather than FastAPI on
purpose: the offline vLLM engine is blocking and single-tenant, so an async framework
would buy nothing but an event loop to block. Run it with
``python -m marathon.server --model Qwen/Qwen3-0.6B``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import reuse_plan
from .canonical import canonical_bytes
from .cold import ColdTier, TransformerEmbedder
from .protocol import BaselineStore, ProtocolError, TurnPayload, resolve_turn
from .remap import Remap, loads_for
from .reuse_plan import _lines
from .session import Session
from .shift_store import churn_tokens

DEFAULT_MAX_TOKENS = 64
#: chat-template markup a message carries beyond its content, for the paging budget
MESSAGE_MARKUP_TOKENS = 8


class ChatTokenizer:
    """Renders a history through the model's chat template, one message at a time.

    The reuse plan needs the token ids *each history entry contributes to the prompt*,
    which a chat template does not hand over directly. They are recovered as the suffix
    the template adds when message ``k`` is appended to messages ``[:k]``. Any preamble
    the template emits before the first message lands inside piece 0, which is why
    ``head_tokens`` is always 0.

    Chat templates are not append-only as written, though: Qwen3 renders a *trailing*
    assistant message with an empty ``<think>`` block and drops it again as soon as
    another message follows, so naive prefix-diffing shifts every later coordinate. Each
    prefix is therefore rendered with a throwaway sentinel message appended, which puts
    every real message in a non-final position and makes the renders genuinely nested.
    The sentinel's own rendered block is derived from the template rather than assumed,
    and the prefix property is checked per message instead of trusted.

    The ids handed to the engine are always a single encode of the full prompt, never
    the concatenated pieces; the pieces supply the *lengths* the plan works in, and a
    disagreement between the two is raised rather than papered over.
    """

    #: appended to every prefix render so no real message is ever the last one
    SENTINEL = {"role": "user", "content": "marathon-sentinel"}

    def __init__(self, model: str, chat_template_kwargs: dict | None = None) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        # Qwen3 is a thinking model; a closed think block keeps the reply short and
        # comparable turn to turn, which is what the per-turn timings measure.
        self.kwargs: dict[str, Any] = {"enable_thinking": False}
        self.kwargs.update(chat_template_kwargs or {})
        # the sentinel's rendered block, read off the template: rendering it twice and
        # subtracting one render leaves exactly one block, preamble excluded
        one = self._render([self.SENTINEL])
        self._block = self._render([self.SENTINEL, self.SENTINEL])[len(one) :]
        if not self._block or not one.endswith(self._block):
            raise ValueError("cannot isolate the sentinel block in this chat template")

    def _render(self, messages: list[dict], generation: bool = False) -> str:
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation, **self.kwargs
            )
        except TypeError:  # template does not accept our kwargs (non-Qwen models)
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation
            )

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _stable(self, messages: list[dict]) -> str:
        """Render ``messages`` with every one of them in a non-final position."""
        text = self._render([*messages, self.SENTINEL])
        if not text.endswith(self._block):
            raise ValueError("chat template did not render the sentinel as a trailing block")
        return text[: -len(self._block)]

    def pieces(self, messages: list[dict]) -> list[list[int]]:
        """Token ids each message contributes to the prompt, in order."""
        out, prev = [], ""
        for k in range(len(messages)):
            cur = self._stable(messages[: k + 1])
            if not cur.startswith(prev):
                raise ValueError(f"chat template is not append-only at message {k}")
            out.append(self.encode(cur[len(prev) :]))
            prev = cur
        return out

    def prompt(self, messages: list[dict]) -> tuple[list[int], list[list[int]]]:
        """``(prompt_ids, pieces)`` for a conversation, generation prompt included."""
        pieces = self.pieces(messages)
        ids = self.encode(self._render(messages, generation=True))
        if sum(map(len, pieces)) > len(ids):
            raise ValueError("per-message token lengths exceed the full prompt")
        return ids, pieces


class VllmEngine:
    """The vLLM offline engine with the Marathon shift connector attached."""

    def __init__(
        self,
        model: str,
        max_model_len: int = 32768,
        gpu_util: float = 0.0,
        store_tokens: int = 16384,
        session_tokens: int = 0,
    ) -> None:
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        self._sampling = SamplingParams
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util or 0.80,
            enable_prefix_caching=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="MarathonShiftConnector",
                kv_connector_module_path="marathon.vllm_shift_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "store_tokens": store_tokens,
                    "session_tokens": session_tokens,
                },
            ),
        )
        self.block_size = self.llm.llm_engine.vllm_config.cache_config.block_size

    def generate(
        self,
        ids: list[int],
        session: str,
        max_tokens: int,
        load: dict | None = None,
        save: bool | str = False,
        loads: list[dict] | None = None,
    ) -> str:
        params = self._sampling(temperature=0, max_tokens=max_tokens)
        kv: dict[str, Any] = {"session": session, "save": save}
        if load:
            kv["load"] = load
        if loads:
            # single-request path: every reused segment at once, non-prefix match
            kv["loads"] = loads
        params.extra_args = {"kv_transfer_params": kv}
        out = self.llm.generate({"prompt_token_ids": ids}, params)
        return out[0].outputs[0].text


class MarathonServer:
    """Verify a turn payload, plan KV reuse, generate, and report what it cost."""

    def __init__(
        self,
        model: str | None = None,
        engine: Any = None,
        tokenizer: Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_model_len: int = 32768,
        gpu_util: float = 0.0,
        store_tokens: int = 16384,
        repair_first: int | None = None,
        reuse: bool = True,
        max_stale: int = 1,
        max_churn: float | None = None,
        max_segments: int = 0,
        gapfill: bool = False,
        resave: bool = True,
        active_window: int | None = None,
        cold_kwargs: dict | None = None,
    ) -> None:
        if engine is None:
            if model is None:
                raise ValueError("MarathonServer needs a model name or an engine")
            # A paged server knows exactly how big a session's KV can get, so the store
            # can allocate it once instead of growing into a GPU with no headroom left.
            cap = active_window + max_tokens + 256 if active_window else 0
            engine = VllmEngine(model, max_model_len, gpu_util, store_tokens, cap)
        self.engine = engine
        self.tok = tokenizer if tokenizer is not None else ChatTokenizer(model)
        self.max_tokens = max_tokens
        self.repair_first = repair_first
        # off = plain vLLM prefix caching, the control an edit turn is measured against
        self.reuse = reuse
        # How many *consecutive* edit turns may be served from re-rotated KV before one
        # is made to recompute honestly. Reused KV is stale by construction -- it
        # attended to the text that preceded it when it was computed -- and an edit turn
        # re-saves the span it just loaded, so without a ceiling the staleness ratchets.
        # Measured 2026-08-19 on a paged session (a front-of-view demotion every turn,
        # 40 turns, Qwen3-0.6B): fact exact-match 0.20 unbounded, 0.50 at 4, and 1.000
        # at 1 against a control of 1.000. An append-only turn needs no reuse and so
        # resets the counter, which is why the default costs Phase 1's pattern nothing:
        # isolated edits separated by ordinary turns are still served entirely from
        # reused KV.
        self.max_stale = max_stale
        # The churn alternative to `max_stale`. Counting *turns* treats every reused edit
        # turn as equally damaging, which it is not: what degrades reused KV is how much
        # of the context it attended to has since been rewritten. When set, the ceiling
        # is instead the accumulated fraction (tokens changed in front of the reused span
        # since its last refresh) / (span length), and `max_stale` is not consulted. A
        # benign repeated edit that barely disturbs the span's prefix can then keep
        # reusing, where the turn counter would have forced a recompute.
        self.max_churn = max_churn
        # Diagnostic: keep at most this many reused segments (the longest ones), leaving
        # the rest for the engine to recompute. Phase 3's HF experiment stitches one
        # cache in a single forward and loses no facts; the connector hands k segments
        # over as k+1 sequential requests. This isolates that difference -- 1 makes the
        # serving path as close to a single stitch as the connector API allows.
        self.max_segments = max_segments
        # Hand every reused segment over in ONE request instead of the k+1 phase
        # sequence, which needs scripts/patch_vllm_gapfill.py applied to vLLM. The
        # phase path is what the 2026-08-21 measurement blamed for the paged workload's
        # answer collapse; this is the alternative it motivates.
        self.gapfill = gapfill
        # ``resave=False`` stops an edit turn re-saving the span it just stitched. The
        # store keeps its generation-0 bytes and a per-session address book
        # (:mod:`marathon.remap`) records how far each span has drifted, so every reuse
        # re-rotates from the original once instead of stitching from stitched values.
        # Testing the compounding hypothesis for the paged collapse (findings 2026-08-21).
        if not resave:
            raise NotImplementedError(
                "resave=False needs a store with per-position validity. ShiftStore is a "
                "flat [base, filled) window whose save at a lower dst_start truncates "
                "everything above it -- which is exactly the generation-0 bytes the "
                "remap addresses. Measured on CPU 2026-08-21: 19 truncating writes and "
                "80 corrupted positions over 20 paged turns. marathon.remap is correct "
                "and tested; wiring it needs an interval-allocating store first. See "
                "docs/single-request-load.md."
            )
        self.resave = resave
        self._remap: dict[str, Remap] = {}
        self.store = BaselineStore()
        self._lock = threading.Lock()
        # per session: previous *active* state and line -> ids cache
        self._prev: dict[str, bytes] = {}
        self._cache: dict[str, dict[bytes, list[int]]] = {}
        self._stale: dict[str, int] = {}
        self._churn: dict[str, int] = {}  # tokens changed in front of the span, since refresh
        # cold tier (Phase 2): None keeps the full history in the active window
        self.active_window = active_window
        self.cold_kwargs = cold_kwargs or {}
        self._cold: dict[str, ColdTier] = {}
        self._full: dict[str, list[dict]] = {}  # previous full history, for the exact trigger
        self._count_cache: dict[str, int] = {}

    def _message_tokens(self, message: dict) -> int:
        """Tokens a message costs the prompt, in the serving layer's own units."""
        text = str(message.get("content", ""))
        n = self._count_cache.get(text)
        if n is None:
            n = self._count_cache[text] = len(self.tok.encode(text))
        return n + MESSAGE_MARKUP_TOKENS

    def cold_for(self, session_id: str) -> ColdTier | None:
        """This session's paging policy, created on first use (None if paging is off)."""
        if self.active_window is None:
            return None
        tier = self._cold.get(session_id)
        if tier is None:
            tier = self._cold[session_id] = ColdTier(
                active_tokens=self.active_window, count=self._message_tokens, **self.cold_kwargs
            )
        return tier

    def plan_for(self, session_id: str, state: bytes, pieces: list[list[int]]):
        """The reuse plan for this session's transition into ``state`` (None if first).

        The tokenizer already produced the ids for every line of the new state, and the
        previous turn's are still cached, so the plan's ``tokenize`` is a dict lookup
        rather than a second pass over the history.
        """
        cache = self._cache.setdefault(session_id, {})
        cache.update(zip(_lines(state), pieces, strict=True))
        prev = self._prev.get(session_id)
        if prev is None or prev == state or not self.reuse:
            return None
        plan = reuse_plan.plan(prev, state, cache.__getitem__)
        if self.repair_first is not None:
            plan = dataclasses.replace(plan, repair_first=self.repair_first)
        return plan

    def turn(self, session_id: str, payload: TurnPayload | dict, generate: bool = True) -> dict:
        """Run one turn for ``session_id``. Returns the reply and per-turn metrics.

        ``generate=False`` runs the whole turn *except* the engine call: the payload is
        still verified, the cold tier still pages, the prompt is still rendered and the
        reuse plan still computed, but nothing is prefilled and ``reply`` comes back
        empty. It exists so a long history can be replayed to bring a session's paging
        state up to date without paying to prefill every turn of it -- the state a
        question is then asked against is identical either way.
        """
        if not isinstance(payload, TurnPayload):
            payload = TurnPayload.from_dict(payload)
        wire_bytes = len(canonical_bytes(payload.to_dict()))
        with self._lock:
            state = resolve_turn(self.store, payload)  # hash-checked; raises otherwise
            messages = Session.decode(state)

            # The cold tier pages the *verified* history down to the active view before
            # anything is tokenized. Demotion and promotion are therefore ordinary edits
            # of the view, and flow through the same reuse plan + shift connector below.
            tier = self.cold_for(session_id)
            events: list = []
            cold_before: list[int] = []
            if tier is not None:
                # what was cold when the turn arrived, before recall acted on it: the
                # eval needs it to tell a recall from a message that never left
                cold_before = sorted(tier.demoted)
                query = payload.new_input or (
                    str(messages[-1].get("content", "")) if messages else ""
                )
                events = tier.step(messages, self._full.get(session_id), query)
                self._full[session_id] = messages
                view, state = tier.view(messages), tier.active_state(messages)
            else:
                view = messages
            ids, pieces = self.tok.prompt(view)

            plan = self.plan_for(session_id, state, pieces)
            loads = plan.to_kv_transfer_params() if plan else []
            if self.max_segments and len(loads) > self.max_segments:
                keep = sorted(loads, key=lambda d: d["dst_end"] - d["dst_start"], reverse=True)
                keep = keep[: self.max_segments]
                loads = [d for d in loads if d in keep]
            stale = self._stale.get(session_id, 0)
            changed, span_len = churn_tokens(loads)
            churn = self._churn.get(session_id, 0) + changed
            churn_frac = churn / span_len if span_len else 0.0
            if self.max_churn is not None:
                refreshed = bool(loads) and churn_frac > self.max_churn
            else:
                refreshed = bool(loads) and stale >= self.max_stale
            if refreshed:
                loads = []  # spend one honest recompute to reset the staleness clock
            # ``planned`` stays in the *plan's* coordinates (delta measured against the
            # previous state); that is what the remap has to be updated with. What the
            # engine gets may be a re-addressed version of it, and feeding those deltas
            # back into the remap would count each offset twice.
            planned = list(loads)
            if not self.resave and loads:
                # reach the generation-0 bytes rather than whatever the last turn
                # stitched; one segment may split into several loads
                loads = loads_for(self._remap.get(session_id, Remap()), loads)
            phases = reuse_plan.phases(loads, self.engine.block_size, len(ids))

            start = time.perf_counter()
            if not generate:
                reply = ""
            elif self.gapfill and loads:
                # one request, every segment: no warm-up phases, so no truncated-prompt
                # prefills and no reliance on the prefix cache to carry earlier segments
                reply = self.engine.generate(
                    ids,
                    session_id,
                    self.max_tokens,
                    loads=loads,
                    save="full" if self.resave else True,
                )
            elif phases:
                # every phase but the last is a max_tokens=1 warm-up whose only job is
                # to leave its blocks in vLLM's prefix cache for the phase after it
                for length, load in phases[:-1]:
                    self.engine.generate(ids[:length], session_id, 1, load=load)
                # "full": the reused span has moved, so the store is rebuilt in the new
                # position coordinates -- otherwise the *next* edit would plan against a
                # layout that no longer exists. See the connector's _plan_save.
                reply = self.engine.generate(
                    ids,
                    session_id,
                    self.max_tokens,
                    load=phases[-1][1],
                    save="full" if self.resave else True,
                )
            else:
                reply = self.engine.generate(ids, session_id, self.max_tokens, save=self.reuse)
            prefill_s = time.perf_counter() - start

            if generate:
                # A non-generating replay turn saves no KV and reuses none, so it can
                # neither ratchet the staleness clock nor reset it. Advancing it here
                # would make a replayed history force spurious refreshes on the real
                # turns that follow (flagged by Track L, 2026-08-19).
                self._stale[session_id] = stale + 1 if loads else 0
                self._churn[session_id] = churn if loads else 0
            if not self.resave:
                prior = self._remap.get(session_id, Remap())
                if planned:
                    segs = [(d["dst_start"], d["dst_end"], d["delta"]) for d in planned]
                    self._remap[session_id] = prior.after_turn(segs, len(ids))
                else:
                    # everything from the first change on was recomputed and re-saved
                    self._remap[session_id] = prior.restrict(plan.p if plan else 0)
            self._prev[session_id] = state
            return {
                "reply": reply,
                "session": session_id,
                "prefill_s": round(prefill_s, 4),
                "prompt_tokens": len(ids),
                "reused_tokens": sum(d["dst_end"] - d["dst_start"] for d in loads),
                "segments": len(plan.segments) if plan else 0,
                "deltas": [seg.delta for seg in plan.segments] if plan else [],
                # where each reused run lands, so a churn metric (how much of what a
                # reused span attended to has since been rewritten) can be computed
                # after the fact rather than needing another run
                "segment_spans": [[s.dst_start, s.dst_end] for s in plan.segments] if plan else [],
                "policy": plan.policy if plan else "first",
                "reason": plan.reason if plan else "first turn of the session",
                "phases": max(len(phases), 1),
                "stale": self._stale.get(session_id, 0),
                "churn": round(churn_frac, 4),
                "remap_depth": self._remap.get(session_id, Remap()).depth(),
                "refreshed": refreshed,
                "wire_bytes": wire_bytes,
                "state_bytes": len(state),
                "active_tokens": len(ids),
                "cold_count": len(tier.demoted) if tier else 0,
                "cold_before": cold_before,
                "promotions": [e.as_dict() for e in events if e.kind == "promote"],
                "demotions": [e.as_dict() for e in events if e.kind == "demote"],
            }


def _handler(server: MarathonServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path != "/v1/turn":
                self._send(404, {"error": "not found"})
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                req = json.loads(raw.decode("utf-8"))
                self._send(200, server.turn(req["session"], req["payload"]))
            except ProtocolError as e:
                # a rejected reconstruction is the client's problem to fix, not a crash
                self._send(409, {"error": type(e).__name__, "detail": str(e)})
            except Exception as e:  # noqa: BLE001 - a bad request must not kill the server
                self._send(400, {"error": type(e).__name__, "detail": str(e)})

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def serve(server: MarathonServer, host: str = "127.0.0.1", port: int = 8000):
    """A ``ThreadingHTTPServer`` exposing ``POST /v1/turn``. The caller runs it."""
    return ThreadingHTTPServer((host, port), _handler(server))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marathon.server", description="Marathon turn server")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-util", type=float, default=0.0)
    p.add_argument("--store-tokens", type=int, default=16384)
    p.add_argument(
        "--repair-first",
        type=int,
        default=None,
        help="natively recompute the first M tokens of each reused segment so it "
        "attends to the new text before it",
    )
    p.add_argument(
        "--max-stale",
        type=int,
        default=1,
        help="consecutive reused *edit* turns before one honest recompute refreshes "
        "the KV (an append-only turn resets the count)",
    )
    p.add_argument(
        "--no-resave",
        action="store_true",
        help="do not re-save a stitched span; keep generation-0 bytes and remap them",
    )
    p.add_argument(
        "--gapfill",
        action="store_true",
        help="single-request multi-segment load (needs scripts/patch_vllm_gapfill.py)",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="diagnostic: stitch at most this many (longest) segments, recomputing the "
        "rest -- 1 approximates a single-shot stitch",
    )
    p.add_argument(
        "--max-churn",
        type=float,
        default=None,
        help="churn ceiling instead of --max-stale: refresh once the tokens rewritten "
        "in front of the reused span exceed this fraction of the span (e.g. 0.2)",
    )
    p.add_argument(
        "--active-window",
        type=int,
        default=None,
        help="cold tier: cap the model-facing window at N tokens, paging out old turns",
    )
    p.add_argument(
        "--no-recall",
        action="store_true",
        help="cold tier control: demote but never promote (stubs only)",
    )
    p.add_argument("--embed-model", default=None, help="cold tier retriever (default: lexical)")
    p.add_argument(
        "--no-reuse",
        action="store_true",
        help="control run: plan nothing, leaving plain vLLM prefix caching",
    )
    args = p.parse_args(argv)
    srv = MarathonServer(
        args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_util=args.gpu_util,
        store_tokens=args.store_tokens,
        reuse=not args.no_reuse,
        max_stale=args.max_stale,
        max_churn=args.max_churn,
        max_segments=args.max_segments,
        gapfill=args.gapfill,
        resave=not args.no_resave,
        repair_first=args.repair_first,
        active_window=args.active_window,
        cold_kwargs={
            "recall": not args.no_recall,
            "embedder": TransformerEmbedder(args.embed_model) if args.embed_model else None,
        },
    )
    http = serve(srv, args.host, args.port)
    print(f"marathon.server ready on http://{args.host}:{args.port}/v1/turn", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        http.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
