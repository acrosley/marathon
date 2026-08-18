# Marathon — Findings

Append-only lab notebook. One dated entry per measured result: command, model, cost, raw numbers, one-line takeaway. Never edit old entries; if something is overturned, add a new entry saying so. [PLAN.md](PLAN.md) says where we are; this file says how we know.

## 2026-08-18 — Live probe: append-only history hits the prompt cache (Phase 0)

Command: `python -m marathon.live_probe --turns 12` · Model: `claude-haiku-4-5` · Cost: ~$0.03

```
turn  ttft_s  input  cache_read  cache_creation
 0    3.19     442        0           0
 1    1.63     867        0           0
 2    1.61    1292        0           0
 3    0.58    1717        0           0
 4    0.77    2142        0           0
 5    2.91    2567        0           0
 6    1.73    2992        0           0
 7    1.40    3417        0           0
 8    1.21    3842        0           0
 9    0.83       3        0        4264   <- first cache write (Haiku minimum ~4k tokens)
10    1.24       3     4264         425   <- reads all prior history, writes only the new turn
11    2.47       3     4689         425
```

Takeaway: once past the cache minimum, unchanged-prefix turns bill 3 uncached tokens; entire history served from cache. Phase 0 exit criterion ("cache-read ≈ total history on unchanged-prefix turns") met.
Caveats: Haiku 4.5 cache minimum is ~4k tokens (not 2k) — run ≥15 turns. TTFT is network-noisy at this scale; no trend claim.

## 2026-08-18 — Live probe: one early edit collapses the cache (Phase 1 motivation)

Command: `python -m marathon.live_probe --turns 16 --edit-at 13` (mutates turn 0 in place at turn 13) · Model: `claude-haiku-4-5` · Cost: ~$0.05

```
turn  ttft_s  input  cache_read  cache_creation
 9    0.85       3     4264           0   <- reads with no writes: leftover hits from previous run (5-min TTL, deterministic filler)
10    0.62       3     4689           0
11    0.50       3     5114           0
12    0.54       3     5114         425
13    1.95       3        0        5968   <- edit: full collapse, whole history re-written
14    0.99       3     5968         425   <- rebuilt
15    0.72       3     6393         425
```

Takeaway: a one-word edit to the first message drops cache reads to zero, re-bills the full history at 1.25× write cost, and roughly 4× TTFT for that turn. Prefix caching is all-or-nothing from the edit point — this is the measured baseline Phase 1 (delta-aware KV reuse) must beat.

## 2026-08-18 — Live probe through the Session runner: same cache behaviour, wire cost flat through the edit

Command: `python -m marathon.live_probe --turns 16 --edit-at 13` · Model: `claude-haiku-4-5` · Cost: ~$0.05
Change: probe now builds every API message from `Session.decode(state)` — the server-verified reconstruction — so the canonical serializer is the only path to the wire. Rows also report Marathon `wire_bytes` vs `state_bytes`.

```
turn  ttft_s  input  cache_read  cache_creation  wire_bytes  state_bytes
 9    0.70       3        0        4264            2332       20254
10    0.59       3     4264         425            2337       22284
11    0.63       3     4689         425            2337       24314
12    0.58       3     5114         425            2337       26344
13    0.64       3        0        5968            2897       28383   <- edit turn
14    0.52       3     5968         425            2303       30413
15    0.52       3     6393         425            2303       32443
```

Takeaway: cache numbers are identical to the hand-built probe, so the earlier findings are a property of the library, not the harness. The contrast the project exists for is visible on turn 13: the provider throws away its cache and re-processes 5,968 tokens, while Marathon's own wire payload absorbs the same edit in 2,897 bytes (~+560 B over steady state) — the delta engine already knows how cheap the edit is; the serving side just can't use that yet. Also: `test_replay_gate.py` now proves in CI that delta-reconstructed state == full-context replay at every turn of a 60-turn session with mid-history edits.

@acrosley 2026-08-18

## 2026-08-18 — Phase 1 stack up: local prefix-cache baseline reproduces the Phase 0 collapse (Qwen3-14B-FP8, vLLM 0.27.1, RTX 5090)

Command: `scripts/phase1_probe.sh --mode prefix --turns 24 --edit-at 20` (WSL2 Ubuntu 24.04, `~/marathon-venv`: torch 2.13.0+cu130, vLLM 0.27.1, LMCache 0.5.3, flashinfer-jit-cache 0.6.16.post3+cu130) · Model: `Qwen/Qwen3-14B-FP8` (15.3 GiB weights, ~10 GiB KV) · Cost: $0, electricity.

Setup notes worth keeping: WSL2 has no UVA, so vLLM 0.27's v2 model runner fails ("UVA is not available") — `VLLM_USE_V2_MODEL_RUNNER=0`. No nvcc in WSL, so FlashInfer's sampler JIT fails — prebuilt `flashinfer-jit-cache` wheel from `flashinfer.ai/whl/cu130`. Model choice: Qwen3.8-27B-FP8 rejected (30.9 GB weights, no KV room; and it is a hybrid Gated-DeltaNet/attention model — 48 of 64 layers carry recurrent state with no per-token KV, so non-prefix KV reuse cannot apply to them). Same for Qwen3.5-4B (24/32 linear). Qwen3-14B is dense full-attention: every layer's KV is a candidate for reuse.

`prefill_s` = wall time of a `max_tokens=1` generate on the offline engine (no network); `prefix_hit_tokens` from vLLM's own `vllm:prefix_cache_hits` counter, per turn.

```
turn  prefill_s  prompt_tokens  prefix_hit  wire_bytes  state_bytes
 17     0.104        10825        10208        3361        54422
 18     0.113        11426        10816        3361        57448
 19     0.117        12027        11408        3361        60474
 20     1.285        12632            0        4524        63509   <- edit turn 0: full collapse
 21     0.122        13233        12624        3396        66535   <- rebuilt
 22     0.122        13834        13216        3396        69561
 23     0.125        14435        13824        3396        72587
```

Takeaway: the provider-side finding is now reproduced on hardware we own, quietly (no network noise: steady-state prefill climbs 0.06 → 0.13 s over 14k tokens; the edit turn is 1.285 s, ~11× the neighbours, with prefix hits at zero), while Marathon's wire payload absorbs the same edit in +1.2 KB. This is the Phase 1 baseline that non-prefix KV reuse (LMCache CacheBlend, `--mode blend`) has to beat on turn 20. Blend mode: engine starts and the blender builds once vLLM's model is registered with LMCache's tracker (`scripts/patch_vllm_blend.py`), but the first store fails in LMCache's pure-torch KV transfer fallback (its compiled `c_ops` doesn't bind against this stack) — being worked.

@acrosley 2026-08-18

## 2026-08-18 — CacheBlend runs end to end, but on LMCache's pure-torch fallback it is slower than no caching

Commands: `scripts/phase1_probe.sh --mode blend --turns 24 --edit-at 20` and `--mode none --turns 24 --edit-at 20` · same stack/model as above · Cost: $0.

Two bugs had to be fixed to get blend past turn 0. (1) The PyPI `lmcache==0.5.3` wheel's compiled `c_ops` extension is ABI-incompatible with torch 2.13 (`undefined symbol: c10::impl::cow::materialize_cow_storage`); LMCache silently falls back to a pure-torch KV transfer whose `single_layer_kv_transfer` has no branch for vLLM 0.27's fused KV layout `[num_blocks, num_kv_heads, block_size, 2*head_size]` → `IndexError`. `scripts/patch_lmcache_fused_kv.py` adds that branch. (2) LMCache strips whitespace from `blend_special_str` and takes `encode(...)[1:]`, so the probe's separator (`encode(" # # ")` = 3 tokens) produced two adjacent split points → a zero-length chunk → `ZeroDivisionError` in the pinned allocator; the probe now derives the separator exactly the way LMCache does. Reuse is real once running: the log shows e.g. `Retrieved 9526 out of 9558 tokens` per turn.

```
turn  prompt_tokens   none_s   blend_s   prefix_s (from previous entry)
 17      10753        1.127     1.169     0.104
 18      11350        1.184     3.307     0.113
 19      11947        1.367     1.258     0.117
 20      12548        1.265     1.364     1.285   <- edit turn 0
 21      13145        1.425     1.647     0.122
 22      13742        1.538     1.686     0.122
 23      14339        1.692     1.824     0.125
```

Takeaway: `none` gives the honest no-cache curve — prefill grows linearly, ~1.3 s at 12.5k tokens, so the prefix-mode edit turn (1.285 s) is exactly "recompute everything". Blend on the torch fallback is no better than that and spikes (3.3 s at turn 18): every layer's KV moves through Python-level gathers to a 20 GB pinned CPU buffer, and the retrieve cost swamps the recompute it saves. Nothing can be concluded about CacheBlend itself from these numbers; the measurement needs the native `c_ops` kernels, which means building LMCache against torch 2.13 (no wheel exists) — needs nvcc, which WSL doesn't have and we have no root for. Being worked (conda/pip-provided CUDA 13.0 toolkit, source build).

@acrosley 2026-08-18

## 2026-08-18 — LMCache built from source against torch 2.13 in 53 s (no root: nvcc ships inside torch's cu13 wheels); blend still bottlenecked on one op

Command: `scripts/phase1_build_lmcache.sh` then `scripts/phase1_probe.sh --mode blend --turns 24 --edit-at 20` · Cost: $0.

No `lmcache` wheel exists for torch 2.13, and WSL has no nvcc and no root for apt — but torch 2.13's cu13 wheels bundle a complete nvcc (13.3.73) under `site-packages/nvidia/cu13/{bin,include,lib}`. `phase1_build_lmcache.sh` symlinks a `~/cuda-home` tree from it (nvcc 13.3 with 13.0 cudart headers trips CCCL's compatibility check → `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`), builds LMCache v0.5.3 from source with `TORCH_CUDA_ARCH_LIST=12.0` in 53 s, and re-applies the venv patches. `c_ops` now binds (no "compiled extension not found" in the log). Except: LMCache's native `single_layer_kv_transfer` — the per-layer op the layerwise CacheBlend path calls — has no branch for vLLM 0.27's fused KV layout either (its `switch` covers formats 0,1,2,3,6,7; the fused 10–13 exist only in the multi-layer kernels), so that one op is unbound back to the patched torch fallback.

```
turn  prompt_tokens  blend_s(native except 1 op)  blend_s(all torch)  none_s  prefix_s
 17      10753          1.081                       1.169              1.127   0.104
 18      11350          1.113                       3.307              1.184   0.113
 19      11947          1.282                       1.258              1.367   0.117
 20      12548          1.329                       1.364              1.265   1.285   <- edit turn 0
 21      13145          1.354                       1.647              1.425   0.122
 22      13742          1.515                       1.686              1.538   0.122
 23      14339          1.619                       1.824              1.692   0.125
```

Takeaway: the spikes are gone and blend is a touch faster than the all-torch run, but the curve is still the linear "recompute everything" line — no edit-turn spike, but no reuse win either, even though LMCache retrieves ~99.7% of tokens every turn (`Retrieved 12505 out of 12547`). The per-layer KV move is still Python. The one thing between this stack and a real CacheBlend number is ~30 lines of CUDA adding the fused branch to `csrc/mem_kernels.cu::single_layer_kv_transfer` (worth upstreaming). Being worked.

@acrosley 2026-08-18

## 2026-08-18 — Fully native CacheBlend ties full recompute; the cost is LMCache's Python layerwise recompute, not KV transfer

Command: `scripts/phase1_build_lmcache.sh` (now applies `scripts/lmcache_fused_single_layer.patch` — 38-line CUDA change adding vLLM 0.27's fused KV formats 12/13 to `single_layer_kv_transfer`; unit-checked bit-exact against a manual gather, HND+NHD, both directions) then `scripts/phase1_probe.sh --mode blend --turns 24 --edit-at 20` · Cost: $0.

```
turn  prompt_tokens  blend_s(native)  none_s  prefix_s
 17      10753          0.757          1.127   0.104
 18      11350          0.795          1.184   0.113
 19      11947          0.845          1.367   0.117
 20      12548          1.260          1.265   1.285   <- edit turn 0
 21      13145          0.981          1.425   0.122
 22      13742          1.301          1.538   0.122
 23      14339          1.530          1.692   0.125
```

LMCache retrieves ~99.7% of tokens every turn (`Retrieved 12505 out of 12547`, vLLM: "External prefix cache hit rate 90.9%") and recomputes a fixed 15% (`blend_recompute_ratios=[0.15]`; the edit changes *which* tokens, not how many). Where an 11.3k-token turn's 0.977 s goes (DEBUG per-layer timestamps): `LMCBlender.blend` 804 ms (82%) — layer 0 over all tokens 174 ms, check layer 34 ms, layers 2–39 over the selected 15% 586 ms (~15 ms/layer) — plus ~170 ms of vLLM prefill for the ~600 uncached tokens and sampling. The KV move itself, measured directly at turn-20 scale (2.06 GB, 40 layers, pinned CPU): 77 ms total, 26.7 GB/s — memcpy speed. So transfer is ~2 ms of the 15 ms per layer; ~90% of blend time is LMCache's own eager Python Qwen3 recompute path at ~9 µs/token/layer vs ~2.5 µs for vLLM's prefill — 3.6× less efficient per token than the prefill it replaces. 0.15 × 3.6 ≈ 0.55, plus two full-length passes and the vLLM tail, lands blend on the recompute-everything line.

Takeaway: on the edit turn CacheBlend-as-shipped is a tie with full recompute (1.260 vs 1.265 s) and it is 7× worse than prefix caching on every unchanged turn (prefix caching is off in blend mode). The idea isn't refuted — with 15% recomputed at vLLM's efficiency the edit turn would be ~0.25 s, a ~5× win — but LMCache 0.5.3's implementation of it can't deliver that on this stack. Phase 1 exit criterion (TTFT win on mid-edit sessions) is NOT met by this route yet. Options, cheapest first: lower `blend_recompute_ratios` (0.05, 0.02) and check output parity against full recompute; blend + vLLM prefix caching on together so unchanged turns keep the 0.12 s path and blend only pays on edits; a late/realistic edit position where prefix caching gets partial hits; or bypass LMCache's recompute path and do the position-shifted KV reuse Marathon's design actually calls for (DESIGN.md "positional entanglement") — the delta engine already knows exactly which byte ranges changed.

@acrosley 2026-08-18

## 2026-08-18 — Position-shifted KV reuse: re-rotated shifted KV holds quality at 0.7% recompute (Qwen3-0.6B, HF)

Command: `scripts/kvshift_probe.sh --model Qwen/Qwen3-0.6B --turns 20` → `marathon.kvshift_probe` (WSL2, `~/marathon-venv`: torch 2.13.0+cu130, transformers 5.15.0, sdpa attention, bf16, no vLLM/LMCache) · Model: `Qwen/Qwen3-0.6B` · Cost: $0.

The route DESIGN.md actually calls for, instead of LMCache: the delta engine says the turn changed one span, so reuse `P` verbatim, compute `E'` fresh, and reuse `S`'s cached V unchanged while **re-rotating** `S`'s cached K by δ = |E'|−|E| positions. RoPE is a rotation, so a key computed at position `p` moves to `p+δ` exactly by one more rotation of angle `δ·θ_i` — no recompute, no approximation. Measured against the real model's `inv_freq`: max abs error 1.3e-05 in fp32 (`tests/test_kvshift.py` proves the identity on a tiny random Qwen3 config, CPU, in CI). What re-rotation cannot fix is that `S`'s KV attended to `E`, not `E'`; that residual is what selective recompute buys back.

Sessions are built with `marathon.session.Session` (20 turns of varied prose, ~5.2k tokens); `marathon.diff` locates the edit (`byte delta head=10783 tail=10599` for the mid edit) and the token span is snapped to it. Four questions per scenario: three planted unique facts (one in `P`, one in the edited span, one in `S`) and one open-ended turn. `klmean` is mean KL vs full recompute over a teacher-forced continuation of the reference's own tokens (stable; free-running greedy agreement is bimodal and is reported but not relied on). `tf_top1` is per-position top-1 agreement. `frac` = tokens forwarded / total; `eff` adds the blend policy's layer-0/1 scan over all of `S`. Worst case over the four questions is shown. `no-rerotate` is the control: same reuse, keys left at their stale angles.

```
scenario    edit          policy         frac    eff   klmean(worst)  tf_top1  QA   prefill_s
edit-turn0  turn 0        full-recompute 1.000  1.000     0.0000       1.00    3/3    0.058
 P=35       E 17->21      no-rerotate    0.007  0.007     0.0168       1.00    3/3    0.024
 S=5152     d=+4          reuse-all      0.007  0.007     0.0033       1.00    3/3    0.023
                          first-32       0.013  0.013     0.0023       0.92    3/3    0.023
                          first-128      0.031  0.031     0.0036       1.00    3/3    0.024
                          blend-r0.05    0.056  0.128     0.0025       1.00    3/3    0.025
                          blend-r0.15    0.155  0.226     0.0022       1.00    3/3    0.035
                          blend-r0.30    0.303  0.374     0.0024       1.00    3/3    0.049
edit-mid    turn 10       full-recompute 1.000  1.000     0.0000       1.00    3/3    0.058
 P=2576     E 18->22      no-rerotate    0.007  0.007     0.0183       0.98    3/3    0.023
 S=2610     d=+4          reuse-all      0.007  0.007     0.0027       0.98    3/3    0.023
                          first-128      0.032  0.032     0.0009       0.98    3/3    0.024
                          blend-r0.15    0.082  0.154     0.0010       0.96    3/3    0.025
                          blend-r0.30    0.157  0.228     0.0006       0.96    3/3    0.035
edit-grow   turn 10       full-recompute 1.000  1.000     0.0000       1.00    3/3    0.134
 P=2576     E 257->466    no-rerotate    0.089  0.089     0.0295       0.92    3/3    0.025
 S=2371     d=+209        reuse-all      0.088  0.088     0.0024       1.00    3/3    0.026
                          first-128      0.112  0.112     0.0019       1.00    3/3    0.027
                          blend-r0.15    0.154  0.226     0.0013       1.00    3/3    0.038
                          blend-r0.30    0.219  0.291     0.0017       1.00    3/3    0.039
```

Takeaway: re-rotated shifted-KV reuse holds up. Recomputing only `E'` plus the new query — 0.7% of tokens on a small edit, 8.8% when the edit grows the history by 209 tokens — keeps mean KL vs full recompute at ~0.002 nats, per-position top-1 agreement at 0.96–1.00, and all planted-fact answers exact. Selective recompute of `S` (CacheBlend-style top-r by layer-1 K deviation, or a flat first-M) lowers KL by a further 2–4× but from an already negligible base; on this workload it does not pay for itself, and the blend policy's layer-0/1 scan over all of `S` costs more (`eff` − `frac` ≈ 0.07) than the recompute it selects. The control settles that re-rotation is doing the work: leaving `S`'s keys at their stale angles is 5–20× worse in KL at identical cost, and is the only policy that ever breaks per-position agreement (0.92). Wall time is 0.023 s vs 0.058 s full recompute (2.5×), but that number is not the claim — a 0.6B model at 5k tokens in HF is launch-latency-bound, not compute-bound, so the honest predictor of a serving win is the recompute fraction: 0.007–0.09 here versus CacheBlend's fixed 0.15 recomputed through LMCache's Python path (previous entry), which tied full recompute.
Caveats: single model size (Qwen3-0.6B); a Qwen3-8B scale-up was started and aborted — the download filled the C: drive (0 bytes free), which wedged the WSL VM; `wsl --terminate Ubuntu` did not bring it back and `--shutdown` is off limits, so WSL needs a manual restart by the lead. ~114 GB was freed from `%TEMP%` (stale diagnostic dumps); the partial Qwen3-8B download is still in the WSL HF cache. Also: in these sessions `S` is semantically independent of the edited span (separate log entries), which is the friendly case for reuse — an edit that later text actually depends on should be the next test.

@acrosley 2026-08-18
