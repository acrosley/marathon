# Marathon — Execution Plan

**Status:** Phase 0 in progress · **Updated:** 2026-08-18 · **Companion:** [DESIGN.md](../DESIGN.md) (doc 0001), [protocol.md](protocol.md)

This plan sequences the delta-encoded context architecture from pure systems work (buildable today against existing APIs) toward the research bet (training the trust contract). Each phase has explicit exit criteria; a phase is not done until its metrics are collected and its correctness gate passes.

## Phase 0 — Deterministic core and prefix-cache maximization *(current)*

Build the ledger, diff engine, and turn protocol; use them to canonicalize state so unchanged history always serializes byte-identically and append-only, making provider prefix caching hit maximally. No custom model, no self-hosted inference.

Delivered so far: canonical serialization with the append-only prefix guarantee, hash-chained ledger with tamper detection, rsync-style block delta engine with randomized round-trip property tests, turn protocol with integrity verification (reconstruction is proven against the target hash, never assumed), offline benchmark harness, experimental live TTFT/cache probe, CI (lint + tests on 3.10 and 3.12).

Remaining for exit: run the live probe to establish real TTFT and cache-read baselines against the Anthropic API; add a session-runner that drives a real conversation through the ledger (canonical serializer as the single path to the wire); publish baseline metrics in-repo (tokens resent per turn, bytes-of-diff vs bytes-of-state, TTFT flat-vs-growing, cache-hit rate); add a full-context replay correctness gate.

**Exit criteria:** wire bytes per turn ~O(|diff| + |input|) in the offline bench (achieved; see README quickstart); live probe shows cache-read tokens ≈ total history tokens on unchanged-prefix turns; correctness replay gate green in CI.

## Phase 1 — Self-hosted inference and warm-tier KV reuse

Stand up vLLM with LMCache to move beyond the prefix restriction: store KV for reused text segments regardless of position and recombine with selective recomputation (CacheBlend-style blending). The delta wire format from Phase 0 becomes the input that decides which KV segments are reusable.

**Exit criteria:** measured TTFT improvement on mid-edit sessions (where prefix caching alone fails) versus the Phase 0 baseline; correctness replay gate still green; documented memory budget per session for hot KV.

## Phase 2 — Cold tier and recall-on-miss

Add the lossy tier: deep history demoted to embeddings/summaries with a paging policy, and recall-on-miss promotion back into the active window when a diff or query touches demoted content. The ledger's content addressing makes promotion exact — cold content is a pointer to verifiable bytes, not a paraphrase.

**Exit criteria:** bounded active-window size on unbounded sessions; recall-on-miss correctly restores demoted content in an eval of targeted questions about old context; quantified quality delta versus full-context replay stays within the agreed tolerance.

## Phase 3 — The trust contract (research bet)

Fine-tune for the contract that absence-from-diff means unchanged: train on delta-formatted interactions with consistency rewards, and evaluate whether the model can safely stop re-reading the baseline without hallucinating or re-requesting it.

**Exit criteria:** on a delta-formatted eval suite, the fine-tuned model matches full-context accuracy within tolerance while consuming only diff + input tokens; documented failure modes and red-team results for baseline-poisoning scenarios.

## Cross-cutting: integrity and operations (never deferred)

Content addressing and hash-chain verification shipped in Phase 0 by design — a trusted substrate is a high-value target, and the model will treat poisoned baselines as settled truth. Still ahead: signed snapshots, single-writer-per-session concurrency (the sane v1 constraint), baseline store TTL/eviction policy, and observability (per-turn metrics exported, not just printed). Every phase keeps the same regression rule: efficiency that changes answers is a regression, not a win.

## North-star metrics

Per-turn prefill tokens → O(|diff| + |input|). TTFT flat as session length grows. Correctness vs full-context replay within tolerance. Cost per turn flat as session length grows.

@acrosley 2026-08-17
