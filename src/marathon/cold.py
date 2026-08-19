"""Cold tier: a paging policy over a Session's history, with recall-on-miss.

The ledger already holds every message content-addressed, so "demoting" one costs
nothing and loses nothing: the bytes stay in the ledger, and only the *active view* --
what actually gets tokenized and prefilled -- swaps the message's text for a compact
stub line::

    [cold #12 3f9a1c04: Here is src/marathon/diff.py: def compute_delta base ...]

That makes demotion an ordinary **edit of the active view** (a shrink), and promotion
the reverse (a grow) -- exactly the shape :mod:`marathon.reuse_plan` already prices
cheaply via position-shifted KV reuse. The cold tier therefore needs no new serving
path; it just moves text in and out of the view and lets the existing machinery
re-rotate everything after the edit.

Two things are never demoted: **governing** spans (the system prompt and any entry
flagged ``governing`` -- editing those is what the reuse plan has to repair, and losing
them changes what steers generation) and the **last K messages** (the working set).

Recall-on-miss has two triggers, per DESIGN.md's "lossy-tier policy" open problem:

``exact``
    the incoming turn's delta touches a demoted message -- an edit landed inside text
    the model can no longer see. The delta engine knows which lines changed, so this is
    detected by comparing old and new canonical lines, not guessed.
``query``
    the new user turn is semantically close to a demoted message. Demoted messages and
    the query are embedded with a small local model (mean-pooled ``all-MiniLM-L6-v2`` by
    default) and the top-k above a similarity threshold come back.

Everything a promotion or demotion does is logged with a reason, because a wrong
demotion degrades the model's ground truth silently -- which is the failure mode the
design doc calls out.

Determinism: the active view is built from the canonical messages by a pure function of
``(messages, demoted set)``, and the stub text is a pure function of the message's index
and its canonical bytes. Same history plus same page-out set always serializes to the
same bytes, so hashes, replay and the reuse plan stay exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_bytes, serialize_history

#: words of the original message kept in the stub, so it stays a usable pointer
STUB_WORDS = 12
#: messages at the end of the history that are never demoted
DEFAULT_KEEP_LAST = 6
#: promotions allowed per turn by the query trigger
DEFAULT_TOP_K = 2
#: cosine similarity a demoted message must beat to be recalled
DEFAULT_THRESHOLD = 0.35
#: words per retrieval chunk, and how much consecutive chunks overlap
CHUNK_WORDS = 60
CHUNK_OVERLAP = 20
#: fallback per-message token estimate (characters per token) when no tokenizer is given
_CHARS_PER_TOKEN = 4
#: chat-template markup every message carries beyond its content
_MSG_OVERHEAD = 8


def is_governing(message: dict[str, Any]) -> bool:
    """Whether a message steers later generation (system prompt, standing instruction).

    Same rule as :func:`marathon.reuse_plan._governing`, read off the message rather
    than off its serialized line.
    """
    return bool(message.get("governing", message.get("role") == "system"))


def stub_text(index: int, message: dict[str, Any], words: int = STUB_WORDS) -> str:
    """The stub line that stands in for a demoted message. Deterministic, byte-stable.

    The hash is over the message's *canonical* bytes, so it is the same content address
    the ledger would give it: the stub is a verifiable pointer, not a paraphrase.
    """
    h = hashlib.sha256(canonical_bytes(message)).hexdigest()[:8]
    head = " ".join(re.split(r"\s+", str(message.get("content", "")).strip())[:words])
    return f"[cold #{index} {h}: {head}]"


@dataclass(frozen=True)
class Event:
    """One paging decision, for the log."""

    kind: str  # "demote" | "promote"
    index: int
    reason: str
    score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {"kind": self.kind, "index": self.index, "reason": self.reason}
        if self.score is not None:
            d["score"] = round(self.score, 4)
        return d


# ------------------------------------------------------------------- embedding


class HashEmbedder:
    """Deterministic bag-of-words embedding. No weights, no GPU, no accuracy claims.

    It exists so the paging policy is testable and runnable on CPU without pulling a
    model down. It is a real (if weak) lexical retriever: hashed unigrams, L2-normalised,
    which recovers exact term overlap -- enough for the planted-fact questions the eval
    asks, but the transformer embedder is the one the numbers are reported for.
    """

    dim = 2048

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class TransformerEmbedder:
    """Mean-pooled last hidden state from a small local encoder.

    Plain ``transformers`` on purpose -- ``sentence-transformers`` is one more dependency
    for one line of pooling, and mean-pooling over the attention mask is exactly what
    MiniLM's sentence-transformer head does.

    CPU by default, deliberately. The retriever shares a process with the serving
    engine, and vLLM is configured to take ~93% of the card; a second CUDA context
    allocating into what is left crashes with ``cudaErrorUnknown`` mid-run (measured
    2026-08-19). MiniLM-L6 is 22M parameters and the paging policy embeds a handful of
    messages per turn, so the GPU buys nothing here and costs a serving outage.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.device = device
        self.max_length = max_length

    def encode(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            enc = self.tok(
                texts[i : i + batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, dim=-1)
            out.extend(pooled.float().cpu().tolist())
        return out


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _chunks(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a message into overlapping windows so one sentence can carry the match.

    Overlapping, so a fact split across a boundary still lands whole in some window.
    """
    parts = text.split()
    if len(parts) <= words:
        return [text] if text.strip() else [""]
    step = max(1, words - overlap)
    return [" ".join(parts[i : i + words]) for i in range(0, len(parts), step)]


# ---------------------------------------------------------------- paging policy


@dataclass
class ColdTier:
    """Keeps the active view under ``active_tokens`` by paging out old messages.

    ``count`` maps a message to the tokens it costs the prompt; pass the serving layer's
    tokenizer so the budget is in the units that actually matter. The default is a
    crude chars/4 estimate, for tests.
    """

    active_tokens: int = 8192
    keep_last: int = DEFAULT_KEEP_LAST
    top_k: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_THRESHOLD
    recall: bool = True
    count: Any = None
    embedder: Any = None
    demoted: set[int] = field(default_factory=set)
    #: demoted indices whose stub was evicted too -- absent from the view entirely
    evicted: set[int] = field(default_factory=set)
    events: list[Event] = field(default_factory=list)
    #: index -> one embedding per chunk, computed once per message and kept (content
    #: never changes while a message is demoted; an exact-trigger edit promotes it first)
    _vectors: dict[int, list[list[float]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.count is None:
            self.count = lambda m: len(json.dumps(m)) // _CHARS_PER_TOKEN + _MSG_OVERHEAD
        if self.embedder is None:
            self.embedder = HashEmbedder()

    # -- view construction (pure) ------------------------------------------

    def stubbed(self) -> set[int]:
        """Demoted indices that still show a stub in the active view."""
        return self.demoted - self.evicted

    def view(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The model-facing active view: demoted messages replaced by their stubs.

        Pure in ``(messages, self.demoted, self.evicted)``. Role and ``governing`` are
        preserved -- only the content changes -- so the view is a valid history and the
        reuse plan sees a plain content edit. An *evicted* message is dropped from the
        view outright; that is a delete edit, which the reuse plan handles the same way.
        """
        stubbed = self.stubbed()
        return [
            {**m, "content": stub_text(i, m)} if i in stubbed else m
            for i, m in enumerate(messages)
            if i not in self.demoted or i in stubbed
        ]

    def active_state(self, messages: list[dict[str, Any]]) -> bytes:
        """Canonical bytes of the active view -- what gets hashed, planned and prefilled."""
        return serialize_history(self.view(messages))

    def active_token_count(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count(m) for m in self.view(messages))

    def protected(self, messages: list[dict[str, Any]]) -> set[int]:
        """Indices that may never be demoted: governing spans and the last K messages."""
        n = len(messages)
        keep = set(range(max(0, n - self.keep_last), n))
        return keep | {i for i, m in enumerate(messages) if is_governing(m)}

    # -- paging ------------------------------------------------------------

    def page(
        self,
        messages: list[dict[str, Any]],
        promote: dict[int, tuple[str, float | None]] | None = None,
    ) -> list[Event]:
        """Bring the active view under budget, honouring ``promote`` first.

        ``promote`` maps an index to ``(reason, score)``. Promoted messages are pinned
        for this call, so if the window then overflows the policy demotes *something
        else* (the oldest eligible message) rather than undoing the recall.

        Returns the events from this call; they are also appended to ``self.events``.
        """
        events: list[Event] = []
        for index, (reason, score) in sorted((promote or {}).items()):
            if index in self.demoted:
                self.demoted.discard(index)
                self.evicted.discard(index)
                self._vectors.pop(index, None)
                events.append(Event("promote", index, reason, score))

        pinned = self.protected(messages) | set(promote or {})
        # oldest first: deep history is the cheapest thing to lose, and the stub keeps a
        # pointer to it, so a wrong choice is recoverable rather than destructive
        eligible = [i for i in range(len(messages)) if i not in pinned and i not in self.demoted]
        used = self.active_token_count(messages)
        for index in eligible:
            if used <= self.active_tokens:
                break
            self.demoted.add(index)
            used = self.active_token_count(messages)
            events.append(Event("demote", index, f"window over budget ({used} tokens after)"))

        # Stubs are ~20x smaller than the messages they replace, but 20x smaller is
        # still O(n): on a long enough session the stubs alone fill the window. When
        # nothing is left to demote and the budget is still blown, the oldest stubs are
        # evicted from the view outright. Nothing is lost -- the bytes are in the ledger
        # and the retriever searches every demoted message, evicted or not, so an
        # evicted message is exactly as recallable as a stubbed one; it just stops
        # costing tokens. This is the step that makes the window genuinely bounded
        # rather than merely 20x smaller.
        for index in sorted(self.stubbed() - pinned):
            if used <= self.active_tokens:
                break
            self.evicted.add(index)
            used = self.active_token_count(messages)
            events.append(Event("demote", index, f"stub evicted ({used} tokens after)"))

        self.events.extend(events)
        return events

    # -- recall-on-miss ----------------------------------------------------

    def touched(
        self, old_messages: list[dict[str, Any]], new_messages: list[dict[str, Any]]
    ) -> set[int]:
        """Demoted indices whose underlying bytes changed -- the exact trigger.

        This is the same comparison the delta engine makes, at line granularity: an edit
        that lands inside a demoted message must bring it back, or the model is asked to
        reason about text it cannot see.
        """
        return {
            i
            for i in self.demoted
            if i < len(old_messages)
            and i < len(new_messages)
            and canonical_bytes(old_messages[i]) != canonical_bytes(new_messages[i])
        }

    def retrieve(self, messages: list[dict[str, Any]], query: str) -> list[tuple[int, float]]:
        """Top-k demoted messages above the threshold for ``query`` -- the query trigger.

        A message is scored by its *best chunk*, not by one vector for the whole thing.
        A history turn is several hundred tokens and the sentence that answers a question
        is one of them; mean-pooling the whole message averages that sentence away, and
        the encoder truncates at 512 tokens anyway, so a fact near the end of a long turn
        is invisible. Measured 2026-08-19: whole-message pooling recalled the right
        message 39% of the time on the Phase 2 question set.
        """
        if not self.recall or not self.demoted or not query.strip():
            return []
        missing = sorted(i for i in self.demoted if i not in self._vectors)
        for index in missing:
            chunks = _chunks(str(messages[index].get("content", "")))
            self._vectors[index] = self.embedder.encode(chunks)
        qvec = self.embedder.encode([query])[0]
        scored = [
            (i, max((cosine(qvec, c) for c in self._vectors[i]), default=-1.0))
            for i in sorted(self.demoted)
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [(i, s) for i, s in scored[: self.top_k] if s >= self.threshold]

    def step(
        self,
        messages: list[dict[str, Any]],
        old_messages: list[dict[str, Any]] | None = None,
        query: str | None = None,
    ) -> list[Event]:
        """One turn of the policy: recall what the turn needs, then page to fit.

        ``old_messages`` is the previous turn's full history (for the exact trigger) and
        ``query`` the incoming user text (for the query trigger). Exact wins: a message
        the delta touched is promoted regardless of the similarity threshold.
        """
        promote: dict[int, tuple[str, float | None]] = {}
        if self.recall:
            if old_messages is not None:
                for i in sorted(self.touched(old_messages, messages)):
                    promote[i] = ("exact: delta touches demoted message", None)
            if query is not None:
                for i, score in self.retrieve(messages, query):
                    promote.setdefault(i, (f"query: similarity {score:.3f}", score))
        return self.page(messages, promote)
