# Marathon — Execution Plan

**Status:** Phase 1 exit criterion met 2026-08-18 (position-shifted KV reuse in vLLM: edit-turn prefill 6.1× faster, output byte-identical); mid-history/grow-edit confirmation and dependent-edit quality study in progress · **Updated:** 2026-08-18 · **Companion:** [DESIGN.md](../DESIGN.md) (doc 0001), [protocol.md](protocol.md), [findings.md](findings.md)

This plan sequences the delta-encoded context architecture from pure systems work (buildable today against existing APIs) toward the research bet (training the trust contract). Each phase has explicit exit criteria; a phase is not done until its metrics are collected and its correctness gate passes.

## Phase 0 — Deterministic core and prefix-cache maximization *(done)*

Build the ledger, diff engine, and turn protocol; use them to canonicalize state so unchanged history always serializes byte-identically and append-only, making provider prefix caching hit maximally. No custom model, no self-hosted inference.

Delivered so far: canonical serialization with the append-only prefix guarantee, hash-chained ledger with tamper detection, rsync-style block delta engine with randomized round-trip property tests, turn protocol with integrity verification (reconstruction is proven against the target hash, never assumed), offline benchmark harness, experimental live TTFT/cache probe, CI (lint + tests on 3.10 and 3.12).

Done 2026-08-18: live probe run against the Anthropic API (cache-read = full history on unchanged-prefix turns; one early edit collapses it — see [findings.md](findings.md)); session runner (`session.py`) with the canonical serializer as the single path to the wire; full-context replay correctness gate in CI (`tests/test_replay_gate.py`); baseline metrics published in README and findings.md. Still open: TTFT flat-vs-growing needs longer sessions than the probe has run; single-writer concurrency and store eviction are tracked under cross-cutting.

**Exit criteria (met 2026-08-18):** wire bytes per turn ~O(|diff| + |input|) in the offline bench; live probe shows cache-read tokens ≈ total history tokens on unchanged-prefix turns; correctness replay gate green in CI.

## Phase 1 — Self-hosted inference and warm-tier KV reuse *(current)*

Started 2026-08-18. Stack: WSL2 Ubuntu on the RTX 5090, `~/marathon-venv` with torch 2.13+cu130, vLLM 0.27.1, LMCache 0.5.3, model `Qwen/Qwen3-14B-FP8` (dense full-attention on purpose — hybrid Gated-DeltaNet models like Qwen3.5/3.8 carry recurrent state with no per-token KV on most layers, so non-prefix KV reuse can't be shown on them; that degradation is a Phase 1.5 question). Reproducible via `scripts/phase1_setup.sh`; probes via `scripts/phase1_probe.sh` → `marathon.local_probe` with modes `none` / `prefix` / `blend`. Local prefix-cache baseline measured (edit-turn collapse: 0.12 s → 1.29 s prefill), CacheBlend now runs fully native, and the route is closed: on the edit turn it ties full recompute at every recompute ratio (0.15/0.05/0.02), it stays 4–5× behind prefix caching on unchanged turns, and blend + prefix caching together crashes inside LMCache. Marathon's own position-shifted KV reuse is the route that works: `kvshift.py` (HF prototype) showed re-rotating cached keys by the edit's token offset δ (RoPE is a rotation, so the shift is exact) and recomputing only the edited span + new query keeps KL vs full recompute ≈0.002 with all planted facts exact at 0.7–9% of tokens forwarded; `vllm_shift_connector.py` (`--mode shift`) puts that inside vLLM 0.27 as a KV connector — on Qwen3-14B-FP8 the edit turn drops from 1.465 s to 0.242 s (6.1×), unchanged turns keep the 0.13 s prefix-cache path, and every turn's output is byte-identical to prefix mode. Open: mid-history and large-δ edits measured on a quiet GPU, edits that later text semantically depends on (where selective recompute must earn its keep), scheduler-safe multi-request connector, eviction.

Original plan text: stand up vLLM with LMCache to move beyond the prefix restriction: store KV for reused text segments regardless of position and recombine with selective recomputation (CacheBlend-style blending). The delta wire format from Phase 0 becomes the input that decides which KV segments are reusable.

**Exit criteria (met 2026-08-18, see findings.md):** measured TTFT improvement on mid-edit sessions (where prefix caching alone fails) versus the Phase 0 baseline — 6.1× on the edit turn; correctness replay gate still green and output parity byte-identical; memory budget for hot KV: 164 KB/token on Qwen3-14B (40 layers × 8 KV heads × 128 × 2 × bf16), 2.7 GB for the 16k-token store buffer.

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
