# Related Work: Non-Prefix / Position-Independent KV-Cache Reuse and Selective Recompute

Survey of 2024-2026 systems and papers on reusing LLM KV cache beyond exact-prefix matching, and on
selectively recomputing only the parts of a context that changed. Written to situate Marathon's
delta-encoded-context approach against the field. All numbers below are as reported by the cited
sources; none are independently verified by Marathon except where explicitly marked "(Marathon
measured)".

## Systems and papers

**CacheBlend** (Yao et al., MSR/Chicago) — Reuses precomputed KV cache for text chunks regardless of
position in the prompt; selectively recomputes a small subset of tokens (chosen by attention-deviation
ranking) to repair cross-attention with new neighboring context. Positions handled via partial
recompute rather than exact re-rotation. Reports TTFT reduced 2.2-3.3x and throughput up 2.8-5x vs.
full recompute, "without compromising generation quality." No fine-tuning required. Code: part of
LMCache (open source).
https://arxiv.org/abs/2405.16444 · https://github.com/lmcache/lmcache

**PromptCache** (Yale/Google, Gim et al.) — Precomputes and stores KV state for reusable "prompt
modules" (system messages, templates, docs) defined via a schema (Prompt Markup Language) with fixed
per-module position IDs; segments are concatenated and reused, not recomputed. Reports up to 8x TTFT
reduction on GPU, up to 60x on CPU, no fine-tuning claimed. Positions handled by pre-assigned module
position IDs, not dynamic re-rotation.
https://arxiv.org/abs/2311.04934

**EPIC** (ICML 2025) — Position-independent context caching (PIC): reuses KV cache chunks regardless of
prefix, via two components — AttnLink (exploits static attention sparsity to limit recompute needed for
accuracy recovery) and KVSplit (semantic-coherence-preserving chunking). Reports up to 8x TTFT and 7x
throughput gains over existing systems with "negligible or no accuracy loss." No fine-tuning stated.
https://arxiv.org/abs/2410.15332

**Block-Attention** (ICLR 2025, Ma et al.) — Divides retrieved documents into independently-encoded
blocks (each computes its own KV state except the final block, which attends across blocks); enables
reuse of previously-seen block KV states in RAG. This method requires fine-tuning the model to attend
correctly under the block-decoupled scheme (unlike training-free schemes above). Specific TTFT/quality
numbers not extracted from search results here — see paper for benchmarks.
https://arxiv.org/abs/2409.15355

**KVLink** (NeurIPS 2025) — Precomputes KV cache per-document independently, then concatenates cached
KV for retrieved documents at inference, addressing the position/cross-attention mismatch this creates.
Reports +4% average QA accuracy over prior SOTA and up to 96% TTFT reduction vs. standard inference.
Code open source.
https://arxiv.org/abs/2502.16002 · https://github.com/UCSB-NLP-Chang/KVLink

**Cache-Craft** (SIGMOD/ACM, 2025) — Manages chunk-caches for RAG; explicitly targets the problem that
current SOTA cannot reuse KV cache when chunks appear at arbitrary positions with arbitrary preceding
context. Stores cross-chunk attention info to preserve accuracy on reuse. TTFT/quality numbers not
extracted here — see paper.
https://arxiv.org/abs/2502.15734

**RAGCache** — Multilevel caching system tailored to RAG retrieval patterns (hot document KV cached in
a cache hierarchy, organized around retrieval frequency). Distinct from CacheBlend/EPIC in focusing on
cache placement/eviction policy for RAG rather than positional recompute. Numbers not extracted here.
https://arxiv.org/abs/2404.12457

**RaaS** — Ambiguous name collision: the paper that resolves cleanly under "RaaS" in 2024-2026 search is
"RaaS: Reasoning-Aware Attention Sparsity for Efficient LLM Reasoning" (attention sparsity for
reasoning-model decoding, not KV-cache reuse/positional recompute). No distinct "Reuse-as-a-Service" or
similarly-named non-prefix KV-cache-reuse paper was found under this acronym — flagging rather than
fabricating a match.
https://arxiv.org/abs/2502.11147

**MPIC** (Feb 2025) — Position-independent multimodal context caching for MLLM serving. Stores KV cache
to local disk on receiving multimodal input, loads/computes in parallel at inference, with an
"integrated reuse and recompute mechanism" to bound accuracy loss. Reports up to 54% response-time
reduction vs. existing context caching, "negligible or no accuracy loss."
https://arxiv.org/abs/2502.01960

**RoPE re-rotation / position-shift work** — Multiple 2025-2026 papers exploit RoPE's closed-form
rotation-composability (`rotate(p1) -> rotate(p2)` via a single delta rotation) to relocate cached keys
to a new position without recomputation, rather than approximate blending:
- **Kamera**: "reuse reduces to exact RoPE re-rotation to any target position, plus a patch that
  restores cross-chunk binding." https://arxiv.org/abs/2606.23581
- **KV Packet**: "recomputation-free context-independent KV caching" using a closed-form RoPE-rotation
  solution to the positional-dependency problem. https://arxiv.org/abs/2604.13226
- **SemPIC**: learns semantic position-independent KV caches. https://arxiv.org/abs/2607.28069
- **Jet-Long**: applies an on-the-fly correction rotation via position offsets instead of full
  recompute, in a long-context-extension setting (not exactly the same problem, but same RoPE-rotation
  mechanism). https://arxiv.org/abs/2607.07740
These are the closest prior art to Marathon's "exact re-rotation, not recompute" idea for the
unchanged-but-shifted-position case; none of them are integrated with a byte-level delta/diff engine
that identifies exactly which spans changed versus merely shifted.

**LMCache** (production system, Sept 2025 tech report) — Pluggable KV cache layer for vLLM: supports
non-prefix KV reuse (any-position block reuse) and integrates CacheBlend for selective recompute on
reuse; storage backends include CPU RAM, local disk, Redis/Valkey, Mooncake, InfiniStore, S3-compatible
object storage, NIXL, GDS. Runs as a standalone daemon so cache survives inference-engine restarts.
Marathon's own local testing (Phase 0, docs/findings.md) used LMCache 0.5.3's native CacheBlend
implementation and found it ties full recompute (1.260s vs 1.265s) at ~12.5k tokens on an edit turn —
i.e., LMCache's reference implementation does not currently deliver the algorithm's paper-reported win
on a vLLM 0.27.1 / Qwen3-14B-FP8 / RTX 5090 stack, because its per-layer eager-Python recompute path is
~3.6x less token-efficient than vLLM's own prefill (Marathon measured).
https://arxiv.org/abs/2510.09665 · https://github.com/lmcache/lmcache

**vLLM automatic prefix caching (APC)** — Production baseline, byte/token-exact prefix match only via
block-hash lookup; no non-prefix or positional-shift handling. Speeds up prefill only, not decode; zero
benefit when a shared prefix doesn't exist or is broken by an early edit. This is the "free" floor every
non-prefix scheme must beat on unchanged turns, and the baseline Marathon measured collapsing (~11x
prefill slowdown, cache reads to zero) on a single early edit.
https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/

**SGLang RadixAttention** — Prefix-cache generalization via a global radix tree shared across all
requests/replicas ever served; still fundamentally prefix-based (branches, not arbitrary-position
reuse). Zheng et al. report up to 6.4x higher throughput vs. no-reuse baselines when prefixes are
shared, and prove the tree-eviction policy is optimal for offline cache-hit-rate. No non-prefix/RoPE
re-rotation capability reported.
https://arxiv.org/abs/2312.07104 (SGLang paper) · https://sgl-project.github.io

**HiCache** — Two distinct things surfaced under this name: (1) SGLang's HiCache, a hierarchical
extension of RadixAttention's radix tree across GPU/CPU/external storage tiers (a cache-placement
system, not a positional/reuse-fraction scheme) — https://www.lmsys.org/blog/2025-09-10-sglang-hicache/
· (2) HiKV (arXiv 2607.22389), an unrelated hardware/algorithm co-design for hierarchical
importance-aware KV *compression* (eviction across two granularities), not cache reuse — reports up to
7.95x attention speedup and 90% energy reduction via a dedicated accelerator. Neither is a
position-independent-reuse or edit-aware system; listed for completeness since "HiCache" was asked for
explicitly.

**Edit-aware / diff-aware reuse for coding assistants and agents** (closest to Marathon's niche):
- **CacheWise** (June 2026) — KVCache management layer for coding agents: prefix-aware scheduling +
  reuse-aware eviction guided by lightweight predictions from tool-call metadata, built on a new
  real-world dataset of coding-assistant sessions (CATraces). Implemented in vLLM; reports 2-2.6x fewer
  cache evictions and up to 3.5x better total agent-session completion time. This is a
  scheduling/eviction-policy contribution, not a byte-diff-driven positional-recompute scheme — it
  decides what to keep/evict, not how to reuse a changed-but-mostly-shared context.
  https://arxiv.org/abs/2606.16824
- **TokenDance** — Multi-agent LLM serving system that implements "diff-aware storage" alongside
  round-aware segment indexing and collective KV-cache reuse; ~3K lines Python + 500 lines CUDA/C++.
  Details on the diff mechanism's granularity (byte-level vs. token/chunk-level) were not resolved from
  search results here.
  https://arxiv.org/abs/2604.03143
- **"Models Take Notes at Prefill: KV Cache Can Be Editable and Composable"** — Directly names the
  problem Marathon targets: "the moment one token changes inside the reused region... the keys and
  values of every later token are invalidated." Proposes making KV cache editable/composable rather
  than solely reuse-or-recompute. Numbers not extracted here — worth a deeper read for Marathon.
  https://arxiv.org/abs/2606.17107
- **Leyline** — "KV Cache Directives for Agentic Inference": lets the agent (which knows context fate
  earlier/more reliably than a system-side predictor) push semantic-edit hints to the cache layer,
  turning eviction from reactive to cooperative. Complementary to Marathon's delta engine (which derives
  exact diffs rather than agent-declared hints) but not the same mechanism.
  https://arxiv.org/html/2606.01065
None of the above combine (a) an exact byte-level diff of what changed with (b) exact RoPE re-rotation
of unchanged-but-shifted spans and (c) recompute limited to genuinely novel/attention-affected spans,
verified against a full-replay correctness gate. CacheWise and TokenDance are the closest in domain
(coding-agent workloads specifically) but operate at scheduling/storage granularity, not exact
positional correction.

## Where Marathon sits

None of the surveyed systems combine all three of: (1) delta-engine-driven, byte-exact identification
of changed spans (not chunk-level, not embedding-similarity-based — Marathon's diff engine already does
this cheaply, per docs/findings.md: ~+560B-1.2KB extra on the wire per edit, vs. providers re-billing
thousands of tokens); (2) exact RoPE re-rotation of unchanged-but-repositioned keys (closest prior art:
Kamera, KV Packet, SemPIC — none integrated with a byte-diff engine, and none applied to a live-serving
edit-turn benchmark against a replay-correctness gate); (3) selective recompute limited only to spans
that are both novel and attention-affected, with the "unaffected but shifted" case handled by exact
re-rotation rather than any recompute or approximate blending. CacheBlend and its descendants
(Cache-Craft, EPIC, MPIC) all handle the position problem via *partial recompute plus quality-repair
heuristics*, not via exact positional correction — the recompute fraction is a knob to be tuned per
workload, not a guarantee. Marathon's design goal is to make that fraction driven by ground-truth diff
output, and to make the "reuse" leg of the tradeoff exact rather than approximate.

Numbers Marathon's own results must beat to claim real progress, since these are the concrete comparison
points now on record:

1. **CacheBlend's own reported numbers**: 2.2-3.3x TTFT reduction and 2.8-5x throughput vs. full
   recompute (https://arxiv.org/abs/2405.16444). This is the algorithm's ceiling as reported by its
   authors on their own hardware/setup — Marathon has not yet matched or approached this on its own
   stack.
2. **vLLm/SGLang prefix caching as the "free" baseline**: near-zero overhead when a request is a byte
   prefix match, and reported up to 6.4x throughput gain (SGLang) when prefixes are shared across
   requests. This is the floor any non-prefix scheme must not regress below on the common case
   (unchanged turns) — Marathon's local finding that CacheBlend's blend mode is 7x worse than prefix
   caching on unchanged turns (because prefix caching is disabled in blend mode) is exactly the failure
   mode to avoid.
3. **Marathon's own Phase 0 finding**: LMCache 0.5.3's native CacheBlend implementation ties full
   recompute on an edit turn (1.260s vs. 1.265s at ~12.5k tokens, RTX 5090 / Qwen3-14B-FP8 / vLLM
   0.27.1) despite marking only 15% of tokens for recompute — because the recompute path itself is 3.6x
   less token-efficient than vLLM's native prefill. This is the number Marathon's own serving-side work
   must clearly beat: matching CacheBlend's reference implementation is not a win; the target is to
   demonstrate a measurable edit-turn speedup over both (a) full recompute and (b) LMCache's own
   CacheBlend timing, using exact re-rotation to eliminate recompute on shifted-but-unchanged spans
   entirely rather than trying to make LMCache's Python recompute path faster.

All of this is unpublished-elsewhere positioning as of the current findings.md/PLAN.md state; no
serving-side implementation delivering these numbers exists yet in this repo.
