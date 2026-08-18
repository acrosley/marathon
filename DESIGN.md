# Marathon — Design Doc 0001: Delta-Encoded Context

**Status:** Draft · **Author:** Andrew Crosley · **Date:** 2026-08-18 · **Scope:** Founding architecture concept for the Marathon project

---

## Summary

Marathon proposes a context architecture in which an LLM is never re-fed its full history. Instead, the model operates against a *deterministic state ledger*: a trusted, precomputed baseline of everything that has not changed, carried as reusable state rather than resent tokens. Each turn transmits only a byte-matched diff plus the new input. Per-turn cost — prefill compute, time-to-first-token, and "startup" reasoning — becomes proportional to what changed, not to total context size. In active-inference terms, the model ingests only prediction error: compute scales with surprise.

## Problem

Today's LLM interaction model is stateless at the wire level. Every turn retransmits and (absent caching) re-processes the entire conversation history, even though the overwhelming majority of that history is byte-identical to the previous turn. This produces three compounding costs. First, prefill compute and latency grow linearly (or worse) with context length, so long-running sessions get slower and more expensive precisely as they accumulate value. Second, the model spends attention re-reading and implicitly re-verifying material it has already processed, which is wasted work and a source of drift. Third, existing mitigations are narrow: provider prompt caching only matches byte-exact *contiguous prefixes*, so a single early edit invalidates everything after it.

## Core idea

Treat the model's context the way rsync treats files and git treats trees: a content-addressed base plus deltas.

**Deterministic history.** The context is a reproducible, append-oriented ledger of states (or stateless snapshots). Determinism is the load-bearing property: because both the client and the serving layer can derive the identical baseline byte-for-byte, the baseline never needs to be retransmitted — only referenced by hash.

**Byte-matched diff overlap.** Unchanged regions are identified by exact byte equality, the same mechanism as rsync rolling checksums or git packfile deltas. Matched regions are referenced, never re-sent, re-tokenized, or re-prefilled. Byte matching is deliberately dumb and cheap; semantic similarity is handled at a different tier (below), not here.

**Tiered representation of the baseline.** The unchanged substrate can be carried in more than one form, trading fidelity against cost:

| Tier | Representation | Fidelity | Cost | Role |
|------|----------------|----------|------|------|
| Hot | Precomputed KV-cache state | Lossless | High memory | Recent window, exact reuse |
| Warm | Blended/recomputed KV segments | Near-lossless | Moderate | Reordered or non-prefix reuse |
| Cold | Embeddings / compressed summaries | Lossy gist | Cheap | Deep history, recall on demand |

**Awareness — the trust contract.** The distinctive commitment of this design: the model receives an explicit guarantee that *anything not present in the diff is unchanged, by definition*. The baseline is a known source of immediate truth. The model does not spend attention re-reading or re-verifying it; absence from the diff is itself load-bearing information. This is as much a training/alignment objective as a systems feature — current models are trained to re-derive everything, and would need to learn to honor the contract.

**Per-turn payload = diff + new input.** The turn protocol sends a baseline reference (hash), the byte-matched delta against it, and the fresh user input. That is the entire active-inference startup cost.

## A turn, end to end

The client computes the current state snapshot and diffs it against the last acknowledged baseline (byte-matched). It sends `{baseline_hash, delta, new_input}`. The serving layer resolves `baseline_hash` to precomputed state — hot KV where available, warm recombination where segments moved, cold embedding recall for deep history — applies the delta, and begins decoding immediately. The model's effective prompt is "here is what changed, here is what's new; everything else is as you knew it." After the turn, the new state is hashed and becomes the next baseline. Ledger determinism means any replica can reconstruct and verify the same state, which also gives you replay, audit, and time-travel debugging for free.

## Prior art and positioning

Provider prompt caching (Anthropic, OpenAI) is the degenerate case of this design: byte-exact matching restricted to contiguous prefixes, because KV entries encode position and attention is not position-independent. Systems like [LMCache](https://github.com/lmcache/lmcache) and its [CacheBlend-style blending](https://docs.lmcache.ai/kv_cache_optimizations/blending.html) push past the prefix restriction by storing KV for arbitrary reused text segments and selectively recomputing the cross-attention needed to stitch them together — reporting roughly 3–10× TTFT improvements in RAG-style workloads. Recent work generalizes further: [adaptive KV-cache reuse for long-context serving](https://arxiv.org/html/2605.24022), [beyond-prefix KV caching](https://arxiv.org/html/2605.07443), and [KV-cache recycling to stretch usable context](https://arxiv.org/html/2512.11851v1). Marathon's contribution is not any single cache trick but the composition: a deterministic ledger as the unit of truth, byte-matched deltas as the wire protocol, tiered state representation, and — the piece no existing system attempts — the explicit trust contract that lets the model treat the unchanged substrate as settled rather than re-derived.

## Open problems

**Positional entanglement.** KV entries are position-dependent; non-prefix reuse requires re-encoding positions or selective recomputation (the CacheBlend approach). This bounds how cheap "warm" reuse can be and is the central systems research question.

**Lossy-tier policy.** Deciding what lives in cold embeddings versus hot KV is a paging problem. A wrong demotion silently degrades the model's ground truth, so the policy needs recall-on-miss (promote cold content back into the active window when the diff or the query touches it).

**Training the trust contract.** Models must learn that absence-from-diff means unchanged. Without training, a model given only a diff will hallucinate or re-request the baseline. This likely requires fine-tuning on delta-formatted interactions with consistency rewards.

**Integrity and security.** A trusted substrate is a high-value target: if an attacker can poison the baseline, the model will treat the poison as settled truth without scrutiny. The ledger needs content-addressing (hashes as identity), signed snapshots, and standard supply-chain hygiene from day one — this is a place to over-invest early, not retrofit.

**Cache coherence in production.** Baselines must be invalidated and versioned like any distributed cache: explicit TTLs, monotonic ledger versions, and a story for concurrent writers (single-writer per session is the sane v1 constraint).

## MVP path (greenfield sequencing)

Marathon does not need a custom model to start. Phase 0 is pure systems work against existing APIs: build the deterministic ledger and byte-matched diff engine, and use them to *maximize provider prefix caching* — canonicalize state so that unchanged history always serializes to an identical prefix, and append-only deltas land after the cache boundary. This alone is measurable (cache-hit rate, TTFT, cost per turn) and forces the ledger design to get real. Phase 1 self-hosts inference and adds non-prefix KV reuse via LMCache/vLLM to unlock warm-tier blending. Phase 2 adds the cold embedding tier with recall-on-miss. Phase 3 — the research bet — fine-tunes for the trust contract and evaluates whether the model can safely stop re-reading the baseline. Instrument everything from Phase 0: tokens resent per turn, bytes-of-diff versus bytes-of-state, TTFT, and a correctness eval that checks the model's answers against full-context replays (the deterministic ledger makes those replays exact).

## Success metrics

Per-turn prefill tokens should approach O(|diff| + |input|) rather than O(|history|). TTFT should stay flat as session length grows. Correctness against full-context replay must remain within an agreed tolerance — efficiency that changes answers is a regression, not a win.

@acrosley 2026-08-17
