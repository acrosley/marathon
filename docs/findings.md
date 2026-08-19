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

## 2026-08-18 — CacheBlend knobs: a lower recompute ratio only speeds *unchanged* turns, never the edit turn; blend + prefix caching crashes

Commands: `scripts/phase1_probe.sh --mode {none,prefix,blend} --turns 24 --edit-at 20 --parity-tokens 16 [--recompute-ratio R] [--blend-prefix]` · Model: `Qwen/Qwen3-14B-FP8`, same WSL2/vLLM 0.27.1/LMCache-from-source stack · Cost: $0.

New probe flags: `--recompute-ratio` (sets `LMCACHE_BLEND_RECOMPUTE_RATIOS`), `--blend-prefix` (blend mode with vLLM `enable_prefix_caching=True` as well), and `--parity-tokens N` — a unique fact (`The access code is 7391-KAPPA.`) is planted in turn 3's user message and the last turn asks `What is the access code? Answer with only the code.` instead of "Reply 'ok'", generating N tokens greedily. Qwen3 is a thinking model and spent the whole budget on `<think>`, so the parity turn's assistant tail prefills a closed empty think block; 16 tokens then suffice. The parity turn is the only one that decodes more than one token, which is why turn 23 is inflated everywhere.

`prefill_s`, turns 17–23 (edit of turn 0 lands on turn 20):

```
turn  prompt_tokens   none   prefix   blend r=0.15   blend r=0.05   blend r=0.02
 17      10766        1.269   0.110      0.686          0.522          0.472
 18      11363        1.399   0.103      0.766          0.557          0.410
 19      11960        1.691   0.098      0.789          0.580          0.521
 20      12561        1.718   1.377      1.399          1.395          1.369   <- edit turn
 21      13158        1.685   0.116      0.878          0.614          0.507
 22      13755        1.884   0.117      0.973          0.651          0.530
 23      14364        1.747   0.210      1.286          1.067          0.830   <- parity turn (16 tokens decoded)
```

Parity: every mode and every ratio answered `7391-KAPPA` exactly — including `none` (ground truth). Dropping the recompute ratio to 0.02 costs no measurable accuracy on this probe.

`--blend-prefix` does not run at all: with vLLM prefix caching on, the scheduler hands LMCache only the *new* tokens after a prefix hit while the blender still assumes the full chunked prompt, so it dies on the first turn with a hit — `RuntimeError: The size of tensor a (1208) must match the size of tensor b (3)` in `blender.process_qkv` (`k` vs `old_k`). Reproduced at r=0.15 and r=0.02; after the crash the engine hangs until LMCache's pin monitor times out at 300 s, so the run has to be killed.

Takeaway: the recompute ratio is the wrong knob — 0.15 → 0.02 buys ~1.7× on unchanged turns (0.79 → 0.52 s at 12k tokens) and *nothing* on the edit turn (1.399 → 1.369 s, still a tie with prefix caching's 1.377 s collapse and with full recompute), because the edit turn's cost is the two full-length passes plus vLLM's prefill of the new tokens, not the selected fraction. Blend still loses 4–5× to prefix caching on unchanged turns, and the "keep both on" escape hatch is closed by an LMCache bug. Of the options listed last entry, the two cheap ones are now spent; the remaining route is Marathon's own position-shifted KV reuse.
Caveats: the box is shared with another agent's GPU work — runs contended by it show 5–40 s spikes and were discarded and re-run (`none` turn 13 at 22.0 s is one survivor; ignore it). A partial `Qwen3-8B` download is still in the WSL HF cache — deleting it was blocked by the permission classifier, so the lead should `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B`.

@acrosley 2026-08-18

## 2026-08-18 — Position-shifted KV reuse inside vLLM: the edit turn drops from 1.46 s to 0.24 s (Qwen3-14B-FP8)

Commands: `scripts/phase1_probe.sh --mode shift --turns 24 --edit-at 20 --parity-tokens 8` and the same with `--mode prefix` · Model: `Qwen/Qwen3-14B-FP8`, same WSL2 stack as the previous entries (vLLM 0.27.1, torch 2.13.0+cu130, `VLLM_USE_V2_MODEL_RUNNER=0`) · Cost: $0.

The HF prototype's re-rotation identity, moved into the serving path. New `src/marathon/vllm_shift_connector.py` is a `KVConnectorBase_V1` (`MarathonShiftConnector`, loaded via `kv_connector_module_path`, `kv_role=kv_both`) that does two things for one session and one writer. SAVE: on every turn it gathers the KV of the tokens vLLM actually computed out of the paged cache into a flat per-layer `[16384, 8, 256]` GPU buffer indexed by absolute position (2.7 GB for 40 layers; `gpu_memory_utilization=0.80` leaves room) — history is append-only, so by the edit turn the buffer holds the previous turn's full KV. LOAD: the request carries `{"load": {"dst_start", "dst_end", "delta"}}` in `SamplingParams.extra_args["kv_transfer_params"]`; `get_num_new_matched_tokens` reports `dst_end - num_computed_tokens` as externally available so vLLM skips prefilling them, and `start_load_kv` copies them in from the buffer — V verbatim, K re-rotated by `delta` (`kvshift.rerotate_keys` as a torch op on vLLM's fused `[num_blocks, num_kv_heads, block_size, 2*head_size]` layout, GPU→GPU, K is `content[:head_size]`).

The probe's edit turn is two phases and `prefill_s` is the sum of both: first a `max_tokens=1` generate of `P + E'` rounded up to a block boundary (native prefill, lands in vLLM's prefix cache), then the real request, on which vLLM prefix-hits `P + E'`, the connector supplies `S` re-rotated, and vLLM prefills only the new turn and the query. On non-edit turns `--mode shift` is plain prefix caching plus the save.

```
turn  prompt_tokens   prefix_s  prefix_hit   shift_s  prefix_hit   text
 17      10766          0.129      10160      0.125      10160     '<think>'
 18      11363          0.124      10752      0.137      10752     '<think>'
 19      11960          0.132      11344      0.128      11344     '<think>'
 20      12561          1.465          0      0.242        640     '<think>'   <- edit of turn 0
 21      13158          0.135      12544      0.138      12544     '<think>'
 22      13755          0.134      13152      0.126      13152     '<think>'
 23      14364          0.236      13744      0.236      13744     '7391-KAPPA'  <- parity question
```

On turn 20 the connector logs `loaded 11328 tokens x 40 layers from store[620:11948], delta=4`: of the 12,561-token prompt, 11,328 tokens (90.2%) are copied-and-re-rotated, 624 are phase-1 prefill of the edited message, and ~610 are the new turn — 9.8% of the prompt forwarded, matching the HF prototype's predictor. **1.465 s → 0.242 s, a 6.1× win on the edit turn**, and the first thing in this project to beat the prefix-cache collapse rather than tie it. Steady-state turns are unchanged (0.12–0.14 s both modes; the per-turn save costs nothing measurable because prefix caching means only ~600 new tokens are ever gathered). Parity: turn 23 answers `7391-KAPPA` in both modes — and it reads that fact through turn 20's re-rotated KV, since turns 21–23 prefix-hit the blocks the connector wrote. Every turn's generated text is byte-identical between the two modes, and identical again on a Qwen3-0.6B bf16 sanity run (10 turns, edit at 8). A sharper version of the parity check — the question asked *on* the edit turn, so the answer can only come from the re-rotated KV that was written moments earlier and never through a later full prefill — also returns `7391-KAPPA` in both modes (Qwen3-0.6B, 9 turns, edit at 8; `loaded 4160 tokens x 28 layers`).

Takeaway: the Phase 1 exit criterion is met on the append-only-plus-one-edit workload — 6.1× TTFT on the edit turn at unchanged output, versus CacheBlend's tie. The mechanism is exactly DESIGN.md's claim: the delta engine knows which byte range moved, RoPE makes the move exact for K, V doesn't care, and only the changed span plus the new tokens need a forward pass.
Caveats, all real: (1) single request, single writer, no eviction, one GPU — this is a probe, not a scheduler-safe connector. (2) The reuse region is copied in whole blocks only, so a sub-block head of `S` is folded into phase 1 and a sub-block tail is recomputed; that is why phase 1 pads up to a block boundary. (3) `S` here is semantically independent of the edited span, the friendly case, same caveat as the HF entry — quality is asserted only by output parity and one planted fact, not by a KL measurement at this scale. (4) The two-phase split is the probe's job, not the connector's; a real server would need the edit's token span from the delta engine directly. (5) A mid-history variant (`--edit-turn 10`, so `P` is 6.5k tokens and `S` is 5.4k) ran correctly — `loaded 5360 tokens x 40 layers from store[6588:11948], delta=4` — but every wall time in that run was contended by another agent's GPU job (steady-state turns at 0.7–5.4 s in two separate attempts), so no timing is quoted from it and the mid-history position remains unmeasured.

@acrosley 2026-08-18

## 2026-08-18 — Shifted KV reuse holds on a mid-history edit (4.7×) and a +186-token grow edit (5.9×); the grow edit flips one token

Commands: `scripts/phase1_probe.sh --mode {shift,prefix} --turns 24 --edit-at 20 --parity-tokens 16` with `--edit-turn 10` (mid-history) and with `--edit-grow 200` (grow) · Model: `Qwen/Qwen3-14B-FP8`, same stack · Cost: $0. New `--edit-grow N` flag prepends filler to the edited message so the edit *adds* ~N tokens instead of shifting by ~4, which is the harder case for re-rotation.

**Mid-history edit** (`--edit-turn 10`: `P` = 6.6k tokens, the connector reuses `S` = 5,360 tokens, `loaded 5360 tokens x 40 layers from store[6588:11948], delta=4`). Prefix caching is *not* fully collapsed here — it hits `P` (5,984 tokens) and recomputes the 6,577 after the edit, so its edit turn is 0.95 s rather than 1.46 s. Shift still wins:

```
turn  prompt_tokens   prefix_s  prefix_hit   shift_s  prefix_hit   text
 17      10766          0.111      10160      0.161      10160     '<think>'
 18      11363          0.114      10752      0.103      10752     '<think>'
 19      11960          0.109      11344      0.098      11344     '<think>'
 20      12561          0.950       5984      0.200      12576     '<think>'   <- edit of turn 10
 21      13158          0.138      12544      0.108      12544     '<think>'
 22      13755          0.134      13152      2.246*     13152     '<think>'
 23      14364          0.231      13744      0.265      13744     '7391-KAPPA'
```

**0.950 s → 0.200 s, 4.7×.** A second shift run reproduced turn 20 at 0.223 s. `*` turn 22 is contended (Track B); so were turns 5/6/15 of that run, and a whole repeat run spiked to 27 s — every quoted row above except the starred one sat in a stretch of clean neighbours, and the prefix run was fully clean (0.076–0.138 s across turns 0–19).

**Grow edit** (`--edit-grow 200` on turn 0: the edited message gains 186 tokens, `loaded 11328 tokens x 40 layers from store[614:11942], delta=186` — a 46× larger shift than the default). Both runs fully clean, no contention:

```
turn  prompt_tokens   prefix_s  prefix_hit   shift_s  prefix_hit   prefix_text  shift_text
 17      10766          0.096      10160      0.098      10160     '<think>'    '<think>'
 18      11363          0.101      10752      0.103      10752     '<think>'    '<think>'
 19      11960          0.097      11344      0.097      11344     '<think>'    '<think>'
 20      12743          1.273          0      0.214        816     '<think>'    '1'        <- edit, delta=+186
 21      13340          0.101      12736      0.102      12736     '<think>'    '1'
 22      13937          0.111      13328      0.110      13328     '<think>'    '<think>'
 23      14546          0.198      13920      0.196      13920     '7391-KAPPA' '7391-KAPPA'
```

**1.273 s → 0.214 s, 5.9×** — the speedup is indifferent to how big the shift is, as expected, since re-rotation is a fixed-cost angle regardless of `delta`. But this is the first output divergence measured: on turns 20 and 21 the single greedy token differs (`'1'` vs `'<think>'`), converging again at turn 22, and the planted-fact answer at turn 23 is still exact in both modes. Turn 21 inherits the difference because it prefix-hits the blocks the connector wrote on turn 20.

Takeaway: the win is robust to edit position (4.7× when prefix caching still gets a partial hit) and to edit size (5.9× at δ=+186). The honest crack is the divergence: the HF prototype predicted this — it measured per-position top-1 agreement at 0.92–1.00, i.e. occasional flips — and at δ=+186 a flip lands on the very first sampled token. The residual is not re-rotation (that is exact) but the fact that `S`'s KV attended to `E`, not `E'`; the prototype showed a first-M or CacheBlend-style selective recompute of `S` cuts the KL 2–4× further, and the connector has no such policy yet. That is the next thing to build, and this measurement is the reason to build it.
Caveats: as the previous entry, plus — one greedy token per turn is a very coarse quality probe; a real check needs teacher-forced KL at 14B scale against a full recompute, which this probe cannot do through vLLM's offline API.

@acrosley 2026-08-18

## 2026-08-18 — Where re-rotated KV reuse actually breaks: instruction spans, not fact spans (Qwen3-8B + 0.6B)

Command: `python -m marathon.kvshift_probe --model Qwen/Qwen3-8B --turns 20` (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, peak 18.5 GiB) · Models: `Qwen/Qwen3-8B` and `Qwen/Qwen3-0.6B` · Cost: $0.

Follow-up to the previous entry, which measured position-shifted KV reuse only where `S` was semantically independent of the edit. Three new scenarios make `S` depend on the edited span: **dep-anaphora** (turn 2 states a code; later turns say "that mission code is from now on the primary key" without repeating the value, so `S` must resolve a reference into `E'`), **dep-instruction** (turn 0 flips a standing "always reply in French" to "in German", which governs the final answer 5k tokens later), **dep-contradict** (an authoritative correction is inserted mid-history revoking a code that a *later* turn still instructs the model to quote). Same metrics as before, plus `==ref`: does the policy produce the same answer as full recompute.

One harness bug had to be fixed first, and it invalidated the dependent scenarios' first run: the probe rendered history as a plain `role: content` transcript, so the model *continued the log* instead of obeying it — at full recompute the 0.6B model ignored the language instruction entirely, which silently turns every instruction-following test into a no-op. The probe now renders through the model's own chat template (`--raw` keeps the old behaviour), and only then does full recompute exhibit the behaviour we are trying to preserve. The independent scenarios were re-run under the template too, so all numbers below are comparable.

```
Qwen3-8B, worst case over each scenario's questions
scenario         policy          frac    eff   klmean   kl_first  tf_top1  QA/lang  ==ref
edit-turn0       full-recompute  1.000  1.000   0.0000    0.0000    1.00     4/4     4/4
 (S independent) reuse-all       0.008  0.008   0.0115    ~0.01     0.94     4/4     3/4
                 first-512       0.104  0.104   0.0067    ~0.01     1.00     4/4     3/4
edit-mid         reuse-all       0.008  0.008   0.0051    ~0.01     0.96     4/4     3/4
edit-grow        reuse-all       0.088  0.088   0.0035    ~0.01     1.00     4/4     4/4
dep-anaphora     full-recompute  1.000  1.000   0.0000    0.0000    1.00     3/3     3/3
 (S -> E' ref)   no-rerotate     0.005  0.005   0.0145    0.0199    0.94     3/3     2/3
                 reuse-all       0.005  0.005   0.0145    0.0199    0.94     3/3     2/3
                 first-512       0.100  0.100   0.0039    0.0199    0.96     3/3     2/3
dep-contradict   full-recompute  1.000  1.000   0.0000    0.0000    1.00     2/2     2/2
 (override)      reuse-all       0.014  0.014   0.0122    0.0579    0.96     2/2     1/2
                 first-512       0.109  0.109   0.0172    0.0414    0.98     2/2     1/2
dep-instruction  full-recompute  1.000  1.000   0.0000    0.0000    1.00    en,en    2/2
 (governing)     reuse-all       0.004  0.004   0.0118    0.3492    0.95    de,de    0/2
                 first-32        0.010  0.010   0.0036    0.0384    0.98    de,en    1/2
                 first-512       0.100  0.100   0.0014    0.0177    0.98    de,en    1/2
                 blend-r0.30     0.300  0.355   0.0051    0.1761    0.95    de,de    0/2
```

Verdict, in two halves. **Fact-level dependence does not break re-rotated reuse.** In dep-anaphora every policy — including plain `reuse-all` at **0.5%** recompute — resolves "that mission code" to the *new* value 9902-SIGMA, exactly as full recompute does; in dep-contradict every policy honours the inserted override and answers 4417-TANGO rather than the revoked code that later text still tells it to quote. The reason is structural: the query attends to `E'` directly, so a fact only has to survive in `E'`, and `S`'s stale hidden states carry the *pointer*, which the edit did not change. First-token KL does rise (0.02 and 0.058, vs ~0.01 in the independent scenarios) — the strain is visible — but not enough to change an answer.

**Governing instructions are where it breaks.** In dep-instruction, `reuse-all` diverges from full recompute on the *first generated token* at KL **0.35** — 30× the ~0.01 of every other scenario — and produces a different language than the reference in 2/2 questions. Recomputing the first tokens of `S` cuts that first-token KL about 10× (0.35 → 0.038 at first-32, → 0.018 at first-512, i.e. 10% of `S`) and restores agreement in half the cases; no policy we tried restores it reliably, and the `blend` selector was the *worst* of the selective options here (0.176 at r=0.30, 35% effective recompute) because it picks tokens by layer-1 K deviation, which the instruction flip barely moves. So the repair is real but partial, and it is bought with an order of magnitude more compute than the fact cases need.

An honest complication on that scenario: the reference is the shakier party. At full recompute the 8B model **ignored** the standing German instruction and answered in English, while the reuse paths answered in German — i.e. the reuse path was arguably the more obedient one. What these runs establish is that reuse *changes behaviour* on a governing-instruction edit; they do not establish which behaviour is correct, and long-context instruction drift at full recompute is a confound we did not control.

For the delta engine the answer is therefore not "mark every downstream span that mentions the edit". Fact-carrying edits need nothing extra — the value is fetched from `E'` on demand. What needs marking is a much narrower class: edits inside spans that *govern* later generation (system/standing instructions, persona, output-format and language directives, tool-use policy). For those, re-rotated reuse should be refused or backed by a recompute of a leading chunk of `S`. That is a cheap classification — such spans are usually the system prompt and the first turn, and the delta engine already knows the byte offsets — and it is a far smaller tax than the dependency-tracking the previous entry's caveat implied.

Scale changed nothing qualitatively. At 8B the independent-scenario picture is the same as at 0.6B (`reuse-all` klmean 0.0035–0.0115, QA 4/4), and dep-anaphora/dep-contradict behave the same in both. Scale does make the speed number meaningful: at 8B and ~5.3k tokens the prefill is 0.42–0.53 s full versus **0.04 s** reused (~11×), 0.08 s at first-512 and 0.19 s at blend-r0.30 — unlike the 0.6B run, this is compute-bound enough that the recompute fraction and the wall clock finally agree.
Caveats: one edit per scenario and 2–4 questions each, so `==ref` counts are small and the open-ended summary question drifts from the reference under *every* policy (greedy agreement 0.06–0.15, and it is no better at 18% recompute) — free-running divergence at 40+ tokens is generic, not a reuse artefact. The instruction scenario tests one instruction type (output language) on one model. Only Qwen3 was tested, and the question tail is built with the model's own chat template, so these numbers are not portable to a model with a different template.

@acrosley 2026-08-18

## 2026-08-18 — `--repair-first` does not fix the grow-edit token flip, and a control shows the flip was never a reuse artefact

Commands: `scripts/phase1_probe.sh --mode shift --turns 24 --edit-at 20 --edit-grow 200 --parity-tokens 16 --repair-first {0,64,256,1024,16384}` and `--repair-first 64` on the default δ=4 edit · Model: `Qwen/Qwen3-14B-FP8` · Cost: $0. GPU quiet throughout (Track B idle); every steady-state turn in every run below sits in 0.063–0.116 s, so no row is contended.

New `--repair-first M` in `--mode shift`. vLLM's connector API can only express externally-matched tokens as a *prefix*, so selective recompute has to be an extra phase, not a scatter: phase 1 prefills `P + E'`, phase 2 prefills `P + E' + S[:M]` (M rounded up to a block multiple) so the head of `S` attends to `E'` instead of the replaced `E`, and the final phase loads only `S[M:]` re-rotated. `prefill_s` sums all phases.

Grow edit (turn 0 gains 186 tokens; `S` = ~11.3k tokens), turn 20:

```
config                     prefill_s  reused_tok  turn20  turn21  turn23(parity)
prefix (no reuse)            1.273         -     '<think>' '<think>'  '7391-KAPPA'
shift, M=0                   0.214      11328     '1'      '1'        '7391-KAPPA'
shift, M=64                  0.240      11264     '1'      '1'        '7391-KAPPA'
shift, M=256                 0.242      11072     '1'      '1'        '7391-KAPPA'
shift, M=1024                0.286      10304     '1'      '1'        '7391-KAPPA'
shift, M=16384 (control)     1.243         16     ' ok'    '<think>'  '7391-KAPPA'
```

Repair costs what you would expect and buys nothing: M=0 → 1024 adds 72 ms (0.214 → 0.286 s, still 4.4× faster than prefix) and the divergent token does not move. The control settles why. M=16384 exceeds `|S|`, so `load_from` clamps to one block before the end and the connector reuses **16 tokens** — that run is a full recompute in all but name, it costs the same as prefix caching's collapse (1.243 vs 1.273 s), and it emits a *third* answer, `' ok'`. Three phase structures over byte-identical token ids give three different first tokens. So turn 20's greedy token here is a near-tie the model resolves at the numerical-noise level, and what flips it is how the prompt is chunked across requests — different batch shapes, different reduction orders in the prefill kernels — not re-rotated KV. The previous entry's "first output divergence measured" reads as a quality signal; it is not one, and this entry overturns that reading. Track B's independent finding the same day — free-running divergence at 40+ tokens is generic and no better at 18% recompute — points the same way.

Default δ=4 edit with `--repair-first 64`: turn 20 is 0.209 s (11,264 tokens reused, `store[684:11948]`) and the text is `'<think>'`, identical to prefix on every turn, as expected — that case never diverged.

Takeaway: `--repair-first` is implemented and measured, and it is not worth its cost on this workload — no measurable quality change at up to 1024 repaired tokens, +34% on the edit turn. That is the same conclusion the HF prototype reached about selective recompute (`eff` − `frac` bought nothing there either), now confirmed in the serving path, with the added serving-specific reason that each repair phase is a whole extra request: M=64 already costs 26 ms, most of it per-request scheduling rather than the 64 tokens of prefill.
Caveats: the near-tie diagnosis rests on a single-token observation on one prompt — a proper statement needs teacher-forced KL against a single-pass reference, which this probe still cannot get through vLLM's offline API. `--repair-first` also cannot express the policy the prototype actually favoured (top-r tokens scattered through `S` by K deviation); the prefix-only connector API rules that out without a second, scatter-shaped load path.

@acrosley 2026-08-18

## 2026-08-18 — Distribution eval, 60 sessions: re-rotated reuse costs 1.6% of tokens for a median KL of 0.0035, and the governing-span rule is right about *where* but wrong about *why*

Command: `scripts/kvshift_eval.sh --model Qwen/Qwen3-8B --sessions 60 --gen-tokens 32 --seed 1234` → new `marathon.kvshift_eval` (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, peak 19.6 GiB, 1297 s) · Model: `Qwen/Qwen3-8B` · Cost: $0.

The previous two entries measured position-shifted reuse on six hand-built scenarios. This is the population version a reviewer would ask for: **60 seeded sessions** (4,378–8,207 tokens, median 6,311) across three families — coding-assistant sessions whose history is real source chunks from this repo, prose sessions built from `docs/*.md`/`DESIGN.md`/`README.md` paragraphs, and Q&A sessions over seeded fact tables — each rendered through Qwen3's chat template. Each session gets **one** edit at a random earlier turn, one of five kinds: `fact` (identifier swap, δ≈0), `rewrite` (body swap, δ −135…+137), `insert` (δ≈+29), `delete` (δ≈−14), and `governing` (the system prompt's standing instruction, δ≈0–1) as its own bucket. Three codes are planted strictly before / inside / after the edited message so the same fact questions are askable under every edit kind. Queries are drawn from a template pool (`fact-before/at/after`, `summarise`, `obey` = "give me a one-line status", `continue-code`) plus, on ~1/3 of items, a question the model wrote itself under full recompute. 200 (session, edit, query) items × 4 conditions.

`klmean` is mean KL over 32 **teacher-forced** continuation tokens of the reference's own greedy output; `kl1` is first-token KL; `tf_top1` per-position top-1 agreement; `exact` is exact match of the 32-token free-running greedy answer against full recompute; `frac` is tokens forwarded. `prefix-equiv` is what vLLM prefix caching can do — reuse `P` only, recompute everything from the edit onward — and it is here for its *cost*, not its quality.

```
overall            n   klmean    klmed    klp95    klmax  kl1mean   kl1max  tf_top1  exact   frac  >.05  >.2
full-recompute   200   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000    1.000   1.00  1.000     0    0
reuse-all        200   0.0142   0.0035   0.0599   0.6382   0.0453   4.1928    0.985   0.64  0.016    11    1
no-rerotate      200   0.0675   0.0128   0.3021   1.5246   0.1479   6.6726    0.969   0.45  0.016    53   18
prefix-equiv     200   0.0017   0.0011   0.0038   0.0452   0.0017   0.0609    0.992   0.80  0.664     0    0
```

```
by edit kind           n   klmean(reuse)  klmean(no-rr)   klp95    klmax  tf_top1  exact    frac
fact                  41      0.0023         0.0025      0.0057   0.0093   0.990    0.76   0.005
insert                41      0.0067         0.0182      0.0245   0.0303   0.988    0.66   0.009
delete                39      0.0115         0.1161      0.0337   0.1352   0.985    0.59   0.004
rewrite               41      0.0167         0.1637      0.0886   0.1419   0.985    0.71   0.058
governing             38      0.0350         0.0372      0.0715   0.6382   0.975    0.45   0.004
```

```
by query type (reuse-all)   n   klmean    klp95    klmax  kl1mean   kl1max  tf_top1  exact
fact-after                 23   0.0048   0.0172   0.0331   0.0021   0.0303    0.995   0.74
summarise                  22   0.0059   0.0154   0.0246   0.0019   0.0290    0.989   0.59
model-written              20   0.0095   0.0479   0.0838   0.1548   2.5864    0.988   0.65
fact-at                    60   0.0114   0.0653   0.0994   0.0004   0.0107    0.988   0.72
continue-code              10   0.0136   0.0457   0.0457   0.0128   0.0687    0.981   0.70
obey                       37   0.0220   0.1137   0.1419   0.1545   4.1928    0.970   0.32
fact-before                28   0.0274   0.0235   0.6382   0.0001   0.0028    0.985   0.79
```

**The claim holds at population scale.** Re-rotated reuse forwards **1.6%** of tokens for a median KL of 0.0035 and a p95 of 0.060; prefix caching's equivalent forwards **66.4%**. 11/200 items clear KL 0.05 and exactly one clears 0.2. Planted-fact retrieval is untouched: 105/111 correct under reuse-all against 106/111 under full recompute — one item's difference, across facts before, inside and after the edit.

**The re-rotation control is now decisive where it can be.** Over the whole population no-rerotate is 4.8× worse in mean KL and puts 53/200 items over 0.05 and 18/200 over 0.2, against reuse-all's 11 and 1, at identical cost. But the per-kind split shows *why*, and it is the useful part: re-rotation only helps where the edit actually shifts positions. On `rewrite` (δ −135…+137) it is a 10× improvement (0.0167 vs 0.1637) and on `delete` (δ≈−14) 10× (0.0115 vs 0.1161); on `fact` and `governing`, where δ ∈ {−1,0,1}, reuse-all and no-rerotate are the same number to three decimals. That is exactly the predicted behaviour of a fixed-angle rotation, measured across 200 items rather than asserted.

**On the governing-span question the rule survives, but its stated mechanism does not.** Governing edits are the worst bucket (klmean 0.0350, exact 0.45, and both the largest KL in the run and the only item over 0.2), and at 19% of items they carry 5 of the 11 over-0.05 items — a 2.4× enrichment. So `reuse_plan`'s refusal is pointed at a real tail. But three things in this data say the reason is not "instructions govern later generation":

1. Governing edits shift nothing (δ ∈ {0,1}), so re-rotation is a no-op there and reuse-all ≡ no-rerotate. Whatever hurts is pure stale attention, not position.
2. The failures land on **fact questions**, not on the instruction-following question. Cross-tabbed: governing × `obey` is 0.0138 mean KL, while governing × every other query is **0.0448**, including the run's worst item (`sid=34`, a governing edit asked `fact-before`, KL 0.638 — with a first-token KL of 0.001, i.e. the divergence is late in the continuation, not at the first token). The 0.35 first-token KL of the hand-built `dep-instruction` scenario does not reproduce anywhere in 60 sessions.
3. The confound is size. The system prompt sits at position 0, so a governing edit leaves `P`≈27 tokens and puts the **entire** history into `S` — `sid=34` has S=7,053. Every token of `S` carries stale attention to the changed span. The other kinds edit a middle turn and leave a real prefix intact. Governing edits are not a semantically special class in this data; they are the class where 100% of the context is downstream of the edit.

The honest restatement is therefore: **the predictor is how much of the context sits after the edit, and "governing" is a proxy for "at the very front".** The two are perfectly confounded here because a standing instruction is always message 0. `reuse_plan` refusing on governing spans still refuses on the right items — it just should not be sold as instruction-awareness, and an edit to a large early *non*-governing turn should be expected to behave the same way. Untested, and the cheapest next experiment.

**The `obey` cell is real but half of it is not reuse's fault.** `obey` is the worst query type for reuse-all — exact 0.32, kl1mean 0.155 with a 4.19 outlier — and the natural read is that a short "one-line status" answer is where a standing instruction bites. But `prefix-equiv` on the same cell only reaches exact 0.59 (0.56 on non-governing × obey), and `prefix-equiv` is mathematically a full recompute of everything that could have changed. So roughly half the disagreement on `obey` is the reference's own instability under a different chunking of the same tokens — the same near-tie effect the `--repair-first` entry diagnosed, now quantified on a population.

**Calibration that a reviewer should hold onto:** `prefix-equiv` scores KL 0.0017 (not 0) and exact-match **0.80** (not 1.00) against a single-pass reference over byte-identical token ids. Free-running 32-token exact match is a metric with a ~20% noise floor on this workload, so reuse-all's 0.64 is to be read against 0.80, not against 1.00 — while the teacher-forced KL, which has no such floor, separates the conditions cleanly (0.0017 / 0.0142 / 0.0675). This is why the KL columns carry the argument and the exact-match column does not.

Takeaway: over 60 realistic sessions and 200 graded items, position-shifted re-rotated reuse diverges from full recompute at a rate of 11/200 above KL 0.05 and 1/200 above 0.2, for 1.6% of the compute prefix caching would spend on the same edits. Re-rotation is what buys that on every edit kind that moves positions. The governing-span rule catches the tail, but the mechanism is span *position and size*, not instruction semantics, and the entry that proposed it (2026-08-18, "instruction spans, not fact spans") is narrowed by this one rather than confirmed.
Caveats: one model (`Qwen3-8B` bf16, sdpa) and one seed — the governing-vs-rewrite ordering rests on 38 vs 41 items and a second seed was not run. Sessions are synthetic, and although their material is real repo text, the turn structure is templated. `continue-code` is n=10 and `governing × obey` is n=12; those cells are indicative only. The three planted codes are the only hard-graded answers; `summarise`, `obey`, `continue-code` and `model-written` are graded solely by agreement with full recompute, which the `prefix-equiv` floor shows is a noisy target. And the position/size-vs-semantics reinterpretation above is an inference from a confounded design, not a controlled result: it needs a run that edits a large early non-governing turn.

@acrosley 2026-08-18

## 2026-08-18 — The 2x2 settles it: the predictor is the governing flag, not the edit's position or |S| — and this overturns the previous entry's reinterpretation

Command: `scripts/kvshift_eval.sh --model Qwen/Qwen3-8B --sessions 84 --gen-tokens 32 --seed 1235` (same WSL2 stack: torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, peak 19.7 GiB, 1675 s) · Model: `Qwen/Qwen3-8B` · Cost: $0.

The previous entry ended by arguing that "governing" was only a proxy for "at the very front, so `S` is the whole history", because a standing instruction is always message 0 and the two were perfectly confounded. That argument was wrong, and this run — the discriminating experiment it asked for — says so. Two new edit kinds break the confound, giving a clean 2x2 with δ held near 0 in all four cells:

* **`early-fact`** — a plain identifier swap in the user turn at position 1 or 2. Front position, huge `S`, **not** governing.
* **`mid-governing`** — the standing instruction is moved out of the system prompt into a mid-history user turn flagged `governing=True` (the system prompt is left neutral, so no unedited copy of the directive survives to contradict the edit), and *that* turn is edited. Mid position, moderate `S`, **is** governing.

Together with the existing `governing` (front, governing) and `fact` (mid, non-governing) that is the full factorial. 84 sessions = 7 edit kinds × 3 families × 4, seed 1235, 269 graded items — which also serves as the **second seed** for the original five kinds.

```
governing flag x edit position (reuse-all)
cell                         n   klmean    klmed    klp95    klmax  exact   meanS  >.05
front, governing            40   0.0264   0.0081   0.0601   0.3676   0.57    6060     4
front, non-governing        39   0.0029   0.0018   0.0100   0.0132   0.87    5652     0
mid, governing              38   0.0255   0.0040   0.1134   0.5133   0.68    3533     3
mid, non-governing          39   0.0020   0.0012   0.0058   0.0110   0.67    3452     0
```

Read down the columns: **the governing flag moves mean KL by ~9x and the position moves it by nothing.** `early-fact` puts 5,652 tokens of stale-attention `S` after the edit — as much as the governing case's 6,060 — and is the *cleanest* cell in the run (0.0029, zero items over 0.05, exact 0.87). `mid-governing` has 3,533 tokens of `S`, barely more than plain `fact`, and is as damaging as a front governing edit (0.0255 vs 0.0264). The previous entry's size hypothesis predicted the exact opposite of both.

**|S| is not the predictor, and there is no threshold to find.** Binned over all 269 reuse-all items, neither |S| nor the downstream fraction is monotone, and the bins that cross p95 = 0.05 are scattered rather than ordered (|S|: `[1000,2000)` and `[4000,5000)`; fraction: `[0,0.2)` and `[0.95,1.01)`). Rank correlations over all items: **spearman(KL, |S|) = −0.028**, **spearman(KL, |S|/prompt) = +0.042** — no relationship in either direction. The one numeric quantity that does correlate is the shift itself, spearman(KL, |δ|) = +0.237.

Conditioning on the flag is what makes the picture snap into focus:

```
NON-GOVERNING only (n=191)                    GOVERNING only (n=78)
|S| bin        n  klmean   klp95  >.05        |S| bin        n  klmean   klp95  >.05
[0,2000)      19  0.0072  0.0152     1        [0,2000)       9  0.0671  0.5133     1
[2000,3000)   48  0.0063  0.0147     0        [2000,3000)    3  0.0038  0.0062     0
[3000,4000)   54  0.0080  0.0385     1        [3000,4000)    7  0.0028  0.0069     0
[4000,5000)   29  0.0066  0.0268     1        [4000,5000)   28  0.0369  0.1978     4
[5000,6000)   12  0.0018  0.0031     0        [5000,6000)    7  0.0073  0.0243     0
[6000,8000)   29  0.0028  0.0100     0        [6000,8000)   24  0.0128  0.0589     2
p95 crosses 0.05 in: no bin                   p95 crosses 0.05 in: 3 of 6 bins
klmean 0.0061  p95 0.0229  >.05 3/191         klmean 0.0260  p95 0.1134  >.05 7/78
>.2 0/191                                     >.2 2/78
spearman(KL,|S|) = -0.162                     spearman(KL,|S|) = +0.058
spearman(KL,|delta|) = +0.407                 spearman(KL,|delta|) = +0.311
```

**Among non-governing edits, p95 KL never crosses 0.05 in any |S| bin or any downstream-fraction bin**, right out to 8k tokens of `S` and a downstream fraction of 0.95+; the correlation with |S| is if anything mildly *negative* (−0.162). Among governing edits it crosses in half the bins with no ordering. Both items in the run above KL 0.2 are governing, and so are the top five items overall — the worst, `sid=69`, is a `mid-governing` edit with **|S| = 1,678 and a downstream fraction of 0.23**, i.e. the single most damaging item in 269 has one of the *smallest* `S` values. A size threshold would not have caught it and would have needlessly refused hundreds of large-|S| fact edits that are fine.

**Seed reproducibility.** The five original kinds rank identically at seed 1235 (n=12 sessions each, as at 1234): fact 0.0020 (was 0.0023), insert 0.0055 (0.0067), delete 0.0062 (0.0115), rewrite 0.0141 (0.0167), governing 0.0264 (0.0350). Overall reuse-all is klmean 0.0119 / median 0.0030 / p95 0.0394 at 1.5% of tokens forwarded, against 0.0142 / 0.0035 / 0.0599 at 1.6% for seed 1234; 10/269 items over KL 0.05 and 2/269 over 0.2. `prefix-equiv` again forwards 68.2% for KL 0.0014 — a 45x cost ratio for a 8.5x KL difference. The `no-rerotate` control is again 6.8x worse (0.0812, 56/269 over 0.05, 25/269 over 0.2), and again the gap is concentrated exactly where δ is large: `rewrite` 0.0141 reused vs **0.3471** unrotated (25x), `delete` 0.0062 vs 0.1550, while `fact`/`early-fact`/`governing`, where δ ∈ {0,1}, show reuse-all ≈ no-rerotate as they must.

**The `obey` cell, revisited.** With `mid-governing` in the mix there are now 58 `obey` items. Governing × obey is klmean 0.0161 / exact 0.46; governing × other queries is **0.0304** / exact 0.70; non-governing × obey is 0.0079 / exact 0.50. So the instruction-following query is *not* where governing edits do their damage — for the second run in a row the damage lands on the fact questions instead. What `obey` does have is a low exact-match under every condition, including `prefix-equiv` at 0.67, which is the reference-instability floor rather than a reuse effect. The mechanism is therefore not "the edited instruction steers the answer"; it is that `S`'s KV attended to the old instruction text and every later token's hidden state is subtly wrong, which shows up wherever the continuation is long enough to accumulate it.

**What `reuse_plan`'s rule should be: exactly what it already is.** Keep the governing flag; do not add a position or |S| threshold, and do not replace the flag with one. Concretely, on this data the current rule refuses 78/269 items and catches both items over KL 0.2 and 7 of the 10 over 0.05; an |S|-based rule with any threshold would refuse far more and catch less. The one honest refinement available is that the flag is *conservative*: 71/78 governing items are under KL 0.05 and would have been fine reused, so `repair` (recompute a leading chunk of `S`) rather than `full` remains the right response — which is what `reuse_plan` already emits. The previous entry's proposed reinterpretation should be treated as retracted.

Takeaway: with the confound broken, the governing flag is the predictor (9x on mean KL, all of the >0.2 tail) and edit position and |S| are not predictors at all — among non-governing edits p95 KL stays under 0.05 in every |S| bin out to 8k. `reuse_plan` keeps its rule unchanged. The 2026-08-18 entry that called "governing" a proxy for "at the front" is overturned by this one; the 2026-08-18 entry before it, which located the failure in governing/instruction spans, is restored — though its *mechanism* (the instruction steering generation) still does not hold, since the damage lands on fact queries, not on the instruction-following query.
Caveats: one model (`Qwen3-8B` bf16, sdpa), two seeds. `mid-governing` is one synthetic construction of "a governing span that is not the system prompt" — a user turn carrying a standing instruction — and real sessions may carry governing content in forms this does not resemble. The 2x2 holds δ near 0 by design, so it says nothing about a *large* governing edit; `rewrite` is the only large-δ kind and it is non-governing. Cell sizes are 38-40 items (12 sessions each), so the two >0.2 items are 2 events and the KL means are not tightly bounded. `continue-code` is n=14. And as before, exact-match has a reference-instability floor (`prefix-equiv` 0.86 overall, 0.67 on `obey`), so the KL columns carry the argument.

@acrosley 2026-08-18

## 2026-08-19 — Multi-span and moved-block KV reuse: cost tracks the number of edits, not the context — and a *relocated* block is not a shifted one

Commands: `scripts/kvshift_probe.sh --model Qwen/Qwen3-8B --turns 20 --scenario multi-k1,multi-k2,multi-k4,multi-k8,move,combined` and `scripts/phase1_multispan.sh` (prefix/shift matrix, `Qwen/Qwen3-14B-FP8`, `--turns 24 --edit-at 20 --parity-tokens 16`) · Cost: $0 · GPU quiet throughout (Track G idle); every steady-state turn quoted below sits in 0.089–0.110 s, so no row is contended.

Every reuse result in this log so far was a *single* edited span, and `reuse_plan` returned `policy="full"` the moment it saw two — a `# ponytail:` ceiling. That is the wrong shape for the workload the design targets: an agent turn rewrites several messages, and blocks get moved. This entry removes the ceiling, and finds that one of the two generalisations works and the other does not.

**The generalisation.** A plan is now a list of `Segment(src_start, src_end, dst_start)`, each carrying its own `delta = dst_start - src_start`; a moved block is a segment whose delta differs from its neighbours' and may be negative. `stitch_segments` places them all (V verbatim, K re-rotated per segment) and everything they do not cover is recomputed in one masked forward, so each fresh span attends to every stitched and freshly written slot below it. The RoPE identity is unchanged; `tests/test_kvshift.py` covers δ ∈ {−7, −1, 0, 1, 13, 64} and `rerotate max abs error` on the real 8B `inv_freq` is 1.39e-05. In the HF prototype `token_segments` runs `marathon.diff`'s rsync matcher over the token-id stream encoded as fixed 4-byte words — the rsync engine is right here precisely because it is *not* an LCS: it indexes every aligned baseline block and matches wherever the bytes occur, so a moved block appears as a copy whose source offset runs backwards, and the 4-byte encoding snaps every match to a token boundary for free. In the serving path `reuse_plan.plan` matches canonical JSONL lines, so a segment is always a whole run of entries.

### Part 1 — quality, Qwen3-8B, 20 turns, ~5.3k tokens, worst case over each scenario's fact questions

`multi-kN` rewrites N different messages in one turn; `move` swaps two messages' contents; `combined` does two rewrites *and* a swap.

```
scenario   segments  reused/total   policy          frac    klmean    klmax  tf_top1   QA   ==ref
multi-k1      3      5288/5326      full-recompute  1.000   0.0000   0.0000    1.00    2/2   2/2
                                    reuse-all       0.012   0.0014   0.0128    1.00    2/2   2/2
                                    no-rerotate     0.012   0.0024   0.0214    1.00    2/2   2/2
multi-k2      6      5276/5345      reuse-all       0.018   0.0012   0.0111    1.00    3/3   3/3
                                    no-rerotate     0.018   0.0024   0.0269    1.00    3/3   3/3
multi-k4      8      5255/5383      reuse-all       0.028   0.0008   0.0095    1.00    4/4   4/4
                                    no-rerotate     0.028   0.0192   0.2256    1.00    4/4   4/4
multi-k8     22      5216/5459      reuse-all       0.049   0.0022   0.0259    1.00    4/4   4/4
                                    no-rerotate     0.049   0.0405   0.3845    1.00    4/4   4/4
move          5      5305/5333      reuse-all       0.009   0.0005   0.0033    1.00    2/2   2/2
                                    no-rerotate     0.009   0.0013   0.0071    1.00    2/2   2/2
combined      -      -              reuse-all       0.028   0.0007   0.0049    1.00    3/3   3/3
                                    no-rerotate     0.028   2.3173  13.5002    0.58    1/3   1/3
```

Deltas are per segment and mixed-sign wherever a block moved: `move` is `[-3689, 0, 14, 3703]`, `multi-k8` runs `[-3001 … 0, 4, 8, 12, 16, 20, 24, 28, 32 … 3193]` across 22 segments. HF prefill, mean over the fact questions: full recompute 0.41–0.45 s against reuse-all 0.039–0.049 s — **9.1× at k=8, 10.1–10.8× elsewhere**, and at 8B this is compute-bound enough that wall clock and recompute fraction agree. Selective recompute again buys nothing: `first-32/128/512` and `blend-r0.05/0.15/0.30` leave klmean in the same 0.001–0.010 band for 2–16× the compute (`first-512` reaches 80% recompute at k=8).

**The control is now decisive.** `no-rerotate` — same segments, keys left at their stale angles — was a 5–20× KL penalty in the single-span entries. With multi-span it becomes a correctness failure: at `combined` it is klmean **2.32**, max 13.5, teacher-forced top-1 agreement 0.58, and 1/3 fact questions right instead of 3/3. At 0.6B the same control literally swaps the two answers in `move` (asked for `alpha` it returns the `omega` code and vice versa). This matches Track G's independent same-day result that the control's damage concentrates where δ is large. Re-rotation is not a refinement; it is what makes non-prefix reuse work at all.

### Part 2 — cost, Qwen3-14B-FP8 in vLLM, 24 turns, edit on turn 20 (12.5k-token prompt)

k segments cannot be handed to vLLM in one shot — its connector API expresses externally-matched tokens only as a *prefix* — so `local_probe._phases` hands them over one per request, each stopping on the block boundary where the next segment begins; the final request is the real one. k edits ⇒ k+1 requests.

```
config                        turn19  turn20  turn21  turn23   req  seg   t20 text / parity        vs prefix
prefix (k=1 mutation)          0.097   1.243   0.107   0.191    1    -    '<think>' / '7391-KAPPA'     -
prefix (pure move)             0.089   1.181   0.109   0.194    1    -    '<think>' / '7391-KAPPA'     -
shift --edit-count 1           0.097   0.198   0.105   0.187    2    2    '<think>' / '7391-KAPPA'   6.3x
shift --edit-count 2           0.099   0.276   0.107   0.193    3    3    '<think>' / '7391-KAPPA'   4.5x
shift --edit-count 4           0.097   0.411   0.105   0.190    5    5    '<think>' / '7391-KAPPA'   3.0x
shift --edit-count 8           0.090   0.664   0.100   0.179    9    9    '<think>' / '7391-KAPPA'   1.9x
shift --move                   0.098   0.293   0.110   0.192    3    5    '<think>' / '7391-KAPPA'   4.0x
shift --edit-count 4 --move    0.095   0.546   0.106   0.188    6    9    '<think>' / '7391-KAPPA'   2.3x
```

Every working row's generated text is byte-identical to prefix mode on every turn, and turn 23 answers the planted fact through re-rotated KV. Steady-state turns are unchanged (0.09–0.11 s in both modes). The edit turn is almost exactly linear in k — 0.198 / 0.276 / 0.411 / 0.664 s is **+66 ms per additional edited message**, which is the k+1-request phase trick plus that message's own prefill, and nothing that scales with context. Extrapolating that fit, shift stops beating prefix caching's flat 1.24 s collapse at about **k ≈ 17 edited messages** in a 12.5k context; below that the win is real, above it a full recompute is simply cheaper.

### A relocated block is not a shifted block

The two `--move` rows above are the *second* version. The first was wrong, and the failure is the most useful thing in this entry.

Transplanting relocated blocks the same way as shifted ones ran fine mechanically — `shift: loaded 576 tokens x 40 layers from store[10777:11353], delta=-10153` and its mirror at `delta=+10154`, no declines, 0.235 s — and produced **garbage**: turn 20 emitted `'1'` instead of `'<think>'`, turn 21 `'2'`, and turn 23 answered `' content used to grow the context in a 2 2 2 2'` instead of `7391-KAPPA`. `--repair-first 256` did not move it (0.340 s, same wrong text), which rules out a seam effect. The cause is the one thing re-rotation cannot touch: a block that moved 10k positions has KV summarising a *completely different* prefix, and re-rotating its keys fixes only where it now sits, not what it attended to. A shift of +4 or +186 leaves the preceding context intact; a relocation does not.

So `reuse_plan` now distinguishes them. A segment whose entries merely shifted is reused; a segment whose entries *relocated* (their index in the history changed) is flagged in `plan.moved` and recomputed instead — `to_kv_transfer_params(reuse_moved=True)` restores the old, measured-unsafe behaviour for anyone who wants to re-measure it. That costs 2 × ~600 tokens of prefill and turns the broken row into the working one: **pure move 1.181 s → 0.293 s (4.0×) with byte-identical output**, `k=4 + move` 1.243 s → 0.546 s (2.3×). Reuse of the *unmoved* 76–95% of the history is untouched. A governing entry that relocated now also trips the `repair` policy, for the same reason a rewritten one does.

This does **not** contradict Part 1, where `move` at 8B scored klmean 0.0005 with both fact questions exact at |δ| ≈ 3.7k. Those questions only ask the model to look up a value living *inside* the moved block, which the query attends to directly — the same structural reason fact edits have always survived. The vLLM probe asks the model to keep generating in context, and that is where a transplanted block's stale summary shows. The honest reading is that the HF move measurement was not sensitive to the failure, not that the two disagree.

### Byte identity is not context identity

The matcher will happily match a block against an *identical passage elsewhere*. It is visible in the segment lists: `multi-k1` contains no move at all, yet its deltas are `[-2385, 0, 4]` — a chunk of the probe's repeated filler prose matched a copy of itself 2,385 tokens earlier. That segment's KV was computed after different preceding text, so it is not merely shifted; it is wrong in the way the move case just demonstrated.

On the HF workload it cost nothing measurable — every scenario above reuses such a segment and still scores klmean ≤ 0.0022 with every fact question exact, for the same "the query attends to the fresh text directly" reason. In the serving path it *did* bite, and the fix is in: `reuse_plan._match` used to take the first unused byte-identical old entry, which mapped a bare `"ok"` acknowledgement to one 10k tokens away (every assistant turn in `local_probe` serialises to the identical line). It now takes the **nearest** candidate, with run-continuation only as a tie-break, so an entry that did not move matches itself and only a genuinely relocated one gets a large delta — `tests/test_reuse_plan.py::test_duplicate_entries_match_the_nearest_not_the_first` pins it. That was not what broke the move case (the relocation itself was — the fix changed which segment one 9-token line belonged to and the output stayed wrong until relocations were recomputed), but it is a real defect found on the way, and it is the same hazard one level down. `token_segments` in the HF prototype remains exposed: the cheap mitigations — require a segment's *predecessor* to match too, or prefer the smallest |δ| among candidates — are not implemented there.

### What changed in the code

`kvshift.Segment` / `token_segments` / `stitch_segments` / `fresh_positions`; `select` and `run_segments` apply first-M and blend per non-leading segment. `reuse_plan.plan` is line-granular, takes `head_tokens` so its coordinates are the serving layer's, and returns `segments` + `moved` + `total`; `to_kv_transfer_params()` returns a **list**, one dict per reused segment past the leading prefix, skipping relocated ones. `policy="full"` now means only "nothing survived" — a truncated history keeps its prefix. `local_probe` derives its plan from `reuse_plan.plan` (its private `_reuse_plan` is gone) and gained `--edit-count k`, `--move` and `--reuse-moved`. `docs/protocol.md`'s reuse-plan section is rewritten. `tests/test_reuse_plan.py` lost `test_two_edits_are_full` — that behaviour is exactly what this entry removes — and gained multi-edit, moved-block (negative δ), duplicate-entry, relocation-policy, `head_tokens` and `_phases` coverage.

Takeaway: **cost scales with the number of edited messages, not with context length** — +66 ms per edit at 12.5k tokens, 6.3× at k=1 down to 1.9× at k=8, against a prefix cache that collapses to 1.24 s regardless — and **quality holds for in-place edits at every k tested**, byte-identical output through k=8 in vLLM and klmean ≤ 0.0022 with every fact answer exact at 8B. Moved blocks are the exception: re-rotation cannot repair a block whose entire preceding context changed, so relocations are recomputed and only shifts reused, which still leaves 4.0× on a pure move.

Caveats: the k-sweep is one session shape (24 turns of ~600-token messages, all edits in the first half) on one model, and the k ≈ 17 break-even is an extrapolation from four points, not a measured crossing. Parity is still one planted fact plus greedy-token equality against prefix mode — the vLLM offline API still cannot give teacher-forced KL at 14B, so Part 1 carries the quality argument and Part 2 carries the cost argument. `combined` at 0.6B *did* break under plain `reuse-all` (answered `5111-SIGMA` for `5111-DELTA`, fixed by first-32); it did not break at 8B, so that failure is not reproduced at scale and may be a small-model artefact. The relocation rule is binary and conservative — it refuses all relocations on the evidence of one |δ| ≈ 10k case, and nothing measures where between δ = 186 (safe) and δ = 10,153 (broken) the boundary sits. And the connector is still one request in flight, one writer, no eviction.

@acrosley 2026-08-19

## 2026-08-19 — North-star: edit-turn TTFT is flat as the session grows (Qwen3-14B-FP8, vLLM 0.27.1)

Command: `scripts/phase1_lengthsweep.sh` → `marathon.local_probe --mode {prefix,shift} --turns T --edit-at T-4 --parity-tokens 16` · Model: `Qwen/Qwen3-14B-FP8` · RTX 5090 / WSL2 · logs `~/marathon-logs/ls_*.{log,json}`

PLAN.md's north-star line is "TTFT flat as session length grows". Every earlier vLLM number was a single 12.5k-token session, which shows a speedup but cannot show a *shape*. This entry sweeps the session length and measures the shape directly. Session turns are the probe's ~597-token filler messages; `--turns` is chosen so the **edit turn** (always last-but-3) lands near each target size, the edit rewrites turn 0, and the final turn asks the planted-fact parity question. `steady` is the mean prefill of the three turns immediately before the edit.

```
edit-turn   turns    prefix          shift            speedup   steady_s      reused   copy
prompt tok           edit_s          edit_s                     prefix/shift  tokens   ms (MB, GB/s)
  4,206      10      0.4193          0.1542            2.7x     0.072/0.074    2,976    24.4 ( 465, 18.6)
  8,382      17      0.6936          0.1544            4.5x     0.082/0.080    7,152    16.0 (1118, 68.4)
 12,561      24      1.2723          0.1951            6.5x     0.100/0.101   11,328    30.2 (1770, 57.2)
 16,143      30      1.7840          0.2006            8.9x     0.113/0.107   14,912    35.5 (2330, 64.1)
 24,501      44      2.9281          0.3612            8.1x     0.139/0.141   23,264   158.4 (3635, 22.4)
 30,471      54      3.8688          0.3663           10.6x     0.156/0.162   29,232   151.6 (4568, 29.4)

mid-history edit (the edited message sits in the middle of the session, not at turn 0)
 16,143      30      0.9550          0.2224            4.3x     0.110/0.117   14,912    15.9 (1118, 68.6)
 30,471      54      2.5715          0.3579            7.2x     0.161/0.163   29,232    61.8 (2235, 35.3)
```

Every row's parity answer is `7391-KAPPA`, and every shift row's generated text is identical to its prefix row on every turn. Each shift edit turn is 2 segments / 2 requests (the `_phases` trick), reusing 2,976–29,232 of the prompt's tokens.

**The claim holds.** Prefix caching's edit turn is a straight line in context length: 0.42 → 3.87 s over 4.2k → 30.5k tokens, a slope of **131 µs per token of history** (r² > 0.99 by eye — the six points are 0.42/0.69/1.27/1.78/2.93/3.87 against 4.2/8.4/12.6/16.1/24.5/30.5k). Shift mode over the same range goes 0.154 → 0.366 s, a slope of **8.1 µs/token — 16× shallower**. At 30k the edit turn costs less than a tenth of what the prefix cache costs, and the gap is still widening; the 24k point (8.1×) dips below the 16k point (8.9×) only because the copy cost stepped up there, not because the prefill did.

**But shift-mode TTFT is not literally flat, and the reason is the copy.** The connector's re-rotate-and-scatter is logged per load now (`copy_ms`, added this run — it was previously untimed). It grows linearly with the reused span: 24 ms at 3.0k tokens rising to 152 ms at 29.2k, i.e. roughly **5 µs per reused token**, which accounts for **essentially the whole 8.1 µs/token shift slope** — 128 ms of the 212 ms that the edit turn gains from 4k to 30k is copy. At 30k the copy is **41% of the entire 366 ms edit turn**. Effective bandwidth is not a clean memcpy either: 57–68 GB/s on the 7k–15k spans but only 22–29 GB/s on the 23k–29k ones, because this is a gathered scatter into paged slots plus a float32 RoPE re-rotation of the K half, not a contiguous copy. So the honest statement is *sub-linear-by-16×, dominated by a memory-bandwidth term that is itself linear in |S|* — not O(1). Making it genuinely flat needs the copy to go away (write re-rotated K in place, or have the attention kernel apply δ at read time), which is a real optimisation and not done.

**Mid-history edits behave the same, from a lower baseline.** When the edit lands in the middle rather than at turn 0, prefix caching keeps the first half as a hit and only collapses from there — 0.955 s at 16k and 2.57 s at 30k, versus 1.78 / 3.87 s for the turn-0 edit. Shift is essentially unchanged (0.222 / 0.358 s), so the speedup is smaller (4.3× / 7.2×) but the *shape* is identical: prefix grows with context, shift does not. This is the expected result — prefix caching's cost is "everything after the earliest edit", so an edit at the midpoint costs about half as much, while shift's cost is "the edited span plus the copy" either way.

**Steady-state prefill is untouched and identical between modes** (0.072 → 0.162 s from 4k to 30k, matching to within noise on every row). That growth is vLLM's own cost of prefilling ~600 fresh tokens against a longer KV cache, and it is the same in both modes — shift mode adds nothing to unchanged turns, which was the point of leaving prefix caching enabled underneath.

Memory: the connector's store is 164 KB/token, so 32k tokens is **5.45 GB** on top of vLLM. The 16k default buffer is now sized per run via a new `--store-tokens` flag (with `--gpu-util`); the 24k and 30k points needed `gpu_memory_utilization 0.78` instead of 0.80 to leave room, peaking at **27.1 GB of 32.6 GB**. Full 32k context does fit — the final turn of the largest run is a 32,274-token prompt — but the *edit* turn tops out at 30.5k because it sits three turns before the end, so the largest edit measured is 30.5k, not 32k.

Caveats: one session shape (uniform ~597-token filler messages), one model, one edit (`--edit-count 1`); the k-dependence from the 2026-08-19 multi-span entry stacks on top of this and is not re-measured here. Six points per line is enough to see a slope, not to bound curvature, and each cell is a single run — the 24k/30k copy-bandwidth drop could partly be run-to-run variance rather than a real size effect. Parity is still one planted fact plus greedy-text equality against prefix mode, not teacher-forced KL. The connector is still single-request, single-writer, no eviction, and the store is a flat position-indexed buffer that must be as long as the session.

Note on process: the first attempt at this sweep died mid-run with a `NameError` because another agent was editing `local_probe.py` in the shared worktree. The runs above were made against a pinned copy of the package (`scripts/phase1_probe_pinned.sh` + `MARATHON_SNAP`, which stages `src/marathon` at git HEAD into `~/marathon-snap` and puts it first on `PYTHONPATH`), so a concurrent edit cannot invalidate a sweep in flight. Only two changes sit on top of HEAD in that snapshot: `--store-tokens`/`--gpu-util` on the probe, and the `copy_ms` timing line in the connector.

@acrosley 2026-08-19

## 2026-08-19 — The shift connector becomes session-keyed and scheduler-safe, and gets *faster*: 14B edit turn 1.24 s → 0.163 s (7.6×)

Commands: `scripts/phase1_sessions_rerun.sh` → `marathon.local_probe --mode {prefix,shift} …` with `--sessions 2` (Qwen3-0.6B) and the standard `--turns 24 --edit-at 20 --parity-tokens 16` (Qwen3-14B-FP8) · same WSL2 stack (vLLM 0.27.1, torch 2.13.0+cu130, `VLLM_USE_V2_MODEL_RUNNER=0`) · GPU quiet (Track H's length sweep finished first) · Cost: $0.

Every connector entry so far ended with the same caveat: *single request in flight, single writer, no eviction* — a probe, not a servable component. This entry removes that caveat, and the honest surprise is that making it correct also made it faster than the version measured in the length sweep an hour earlier.

**What the connector is now.** The flat "last request's KV" buffer is gone. `marathon.shift_store` holds a `ShiftStore` keyed by session id (taken from `kv_transfer_params["session"]`), one position-indexed buffer per session per layer, and a `SessionTable` that enforces one in-flight writer per session — the v1 concurrency rule DESIGN.md and protocol.md already assumed. A request with no session id is pass-through: no load, no save, vLLM behaves as if no connector were configured. A second concurrent request on the same session is refused reuse and logged, which also means a load can never overlap an in-flight save. Saves are per scheduler step from `num_computed_tokens`, so chunked and continued prefills are recorded correctly, and a save at a lower `dst_start` truncates the stale positions above it — which is exactly what an edit is. The store carries a total token budget (`MARATHON_STORE_TOKENS`, default 32768; 164 KB/token on Qwen3-14B) with LRU eviction of whole sessions. Because eviction happens on the worker while the *scheduler* is the side that promises vLLM a span needs no prefill, the scheduler-side connector runs the same bookkeeping without tensors and declines a load whose source positions were evicted or truncated: a miss costs a recompute, never a wrong answer. `stats()` (tokens per session, hits, misses, evictions, refusals, loads, saves) is logged from the engine-core process, since vLLM spawns it and the caller cannot reach the object.

**Two interleaved sessions, both edited, prove isolation.** `local_probe --sessions 2` runs N independent `Session` objects turn by turn (s0 turn 0, s1 turn 0, s0 turn 1, …), each planting a *different* access code at turn 3 and each editing turn 0 on turn 9; a store that leaked across sessions would answer with the other session's code.

```
Qwen3-0.6B, 12 turns, edit on turn 9, two sessions, --store-tokens 32768
turn  session   prefix_s   shift_s   reused       parity (both modes)
  8      s0      0.0126    0.0142      -
  8      s1      0.0131    0.0146      -
  9      s0      0.0578    0.0258    4800/6034     <- edit turn, 2 segments / 2 requests
  9      s1      0.0595    0.0249    4800/6034
 10      s0      0.0152    0.0139      -
 10      s1      0.0149    0.0144      -
 11      s0      0.0291    0.0289      -           '7391-KAPPA'
 11      s1      0.0289    0.0285      -           '5820-OMEGA'
```

All 24 rows (12 turns × 2 sessions) are byte-identical between prefix and shift mode, and each session answers **its own** code. The store ends at `{'s0': 5414, 's1': 5414}` tokens with `hits: 2, misses: 0, evictions: 0, refusals: 0` — both edit turns reused, neither session disturbed the other. The edit turn is 2.3× at this (tiny) model size.

**The 14B regression check came out ahead of the old probe.**

```
Qwen3-14B-FP8, 24 turns, edit of turn 0 on turn 20, --store-tokens 16384
turn   prompt_tokens   prefix_s   shift_s   prefix_hit(shift)   text
 17       10,766        0.0948     0.0960       10,160          '<think>'
 18       11,363        0.0990     0.1001       10,752          '<think>'
 19       11,960        0.0951     0.1214       11,344          '<think>'
 20       12,561        1.2354     0.1629          640          '<think>'   <- edit turn
 21       13,158        0.1061     0.1061       12,544          '<think>'
 22       13,755        0.1056     0.1066       13,152          '<think>'
 23       14,364        0.1878     0.1858       13,744          '7391-KAPPA'
```

**1.2354 s → 0.1629 s, 7.6×**, with all 24 turns' generated text byte-identical to prefix mode and steady-state turns matching prefix mode to within noise. The same session and edit shape measured 0.242 s (6.1×) in the 2026-08-18 entry and 0.195 s in this morning's length sweep, so the scheduler-safe rewrite is not a cost — it is the fastest number this workload has produced.

**Why it got faster: the load copy was never really 57 GB/s.** The length-sweep entry attributed most of shift's residual slope to the re-rotate-and-scatter copy (30 ms / 57 GB/s at this size, 152 ms at 30k). Getting the rewrite wrong twice exposed what that number actually measures. The first version grew each session's buffers in 1024-token steps, reallocating all 40 layer tensors every other turn: the copy fell to **4.7 GB/s (364 ms)** and *every* turn in the run slowed 2–4× (steady state 0.21–0.41 s against prefix mode's clean 0.095–0.106 s in the same session — the tell that it was allocator churn, since turns 21–23 run no connector code at all). Doubling instead of stepping recovered most of it (16.7 GB/s, 104 ms, edit turn 0.265 s). Allocating a session's buffers **once as a slab** (`SLAB = 16384` positions, capped at the budget, doubling only if a session outgrows it) took the same 1770 MB copy to **3.15 ms — 549 GB/s**, and the two-session 0.6B loads to 211 and 311 GB/s. So the copy is not memory-bandwidth-bound at all when the store is one contiguous per-layer block; the 22–68 GB/s figures in the sweep were measuring fragmentation of a store that had been allocated in pieces. That does not overturn the sweep's *shape* argument — the copy is still linear in the reused span — but it moves the constant by more than an order of magnitude, and the sweep's "copy is 41% of the edit turn at 30k" should be re-measured against the slab store before it is believed.

**A second bug the tests did not catch, which only a run could.** The first rewrite required a session's stored positions to be contiguous from 0 and refused any save that would leave a hole. Every save was then refused (`refusals: 9`, store empty, load declined, silent fall back to full recompute — parity still correct, speedup gone), because a request only ever computes what vLLM's prefix cache did *not* already have: a session's first save starts at its prefix hit, not at position 0. The store now records a `base` as well as a high-water mark and never claims to hold the head that vLLM serves itself. `tests/test_shift_store.py::test_the_first_save_may_start_above_zero` pins it.

**What changed in the code.** New `src/marathon/shift_store.py` (`slots`, `SessionTable`, `ShiftStore`), pure torch and testable on CPU with fake KV — 18 tests covering the single-writer guard, position bookkeeping (append, hole refusal, edit truncation, base), budget/LRU eviction, session isolation, growth preserving data, and the scheduler/worker bookkeeping mirror. `vllm_shift_connector` is rewritten around it and keeps the `copy_ms` instrumentation. `local_probe` gained `--sessions N` (the session tag is only added to prompts when N > 1, so single-session runs stay byte-comparable with every earlier entry) and prints the store stats. `docs/protocol.md` gained a "Connector" section listing what is now safe and what is not.

Takeaway: session-keyed, budgeted, single-writer-enforced and eviction-aware costs nothing in speed — the edit turn is 7.6× at 12.5k tokens with byte-identical output, and two interleaved sessions each keep their own facts. The store's *allocation shape*, not its bandwidth, was the dominant cost in the load path.

Caveats, all real: tensor parallelism is untested (each worker would keep its own store and nothing coordinates them). Preemption is untested — a request preempted and recomputed mid-flight keeps its store positions, and nothing has measured whether the re-issue lands on the same coordinates. Chunked prefill *interleaved with* a load on the same request is untested; the probe only ever hands a load to a request whose local prefix hit already ends at `dst_start`. The scheduler/worker mirror stays in step because both sides see the same saves in the same order, which holds for the probe's sequential requests but has not been stress-tested against a scheduler that retries `get_num_new_matched_tokens` many times or reorders requests — under real concurrency the two LRU orders could drift, and the failure mode is a declined load, not a wrong one. Eviction under pressure is covered only by CPU tests, never by a GPU run that actually evicted a live session. The slab wastes memory on short sessions (a 4k session still takes a 16k slab), and two 14B sessions at the default budget are 5.4 GB. Reuse is still whole-block only, and parity is still one planted fact plus greedy-text equality against prefix mode.

@acrosley 2026-08-19

## 2026-08-19 — A fused Triton kernel takes the shifted copy to memcpy speed (1.43 TB/s), and the 30k edit turn drops to 0.26 s

Commands: `scripts/bench_shift_copy.sh` (micro-benchmark) and `scripts/bench_shift_30k.sh` (the 30k point of the length sweep, both modes, against a pinned snapshot) · Model: `Qwen/Qwen3-14B-FP8` · RTX 5090 / WSL2, torch 2.13 + Triton 3.7.1 · logs `~/marathon-logs/{bench_shift_copy,ls___mode_*_turns_54*}.log`

The previous entry left the connector's re-rotate-and-scatter as the whole slope of the shift-mode edit turn: ~5 µs per reused token, 22–68 GB/s, 41% of the 0.37 s edit turn at 30k. In torch that copy is five passes over the same bytes — slice K, upcast to fp32, materialise `rotate_half` as a fresh allocation, fuse, downcast, `cat` K and V back together, then an advanced-indexing scatter. `src/marathon/shift_kernels.py` replaces it with one Triton pass per layer: read each source token's `[K|V]` row out of the session store, rotate K in-register in fp32, write it straight into its destination block slot in the paged layout. Every byte is read once and written once.

The rotation is folded into two tables so the kernel has no branches: `out[i] = k[i]·cos[i] + k[partner[i]]·sgn[i]`, where `partner` is `rotate_half`'s (i, i+d/2) pairing and `sgn` carries its sign flip. That is the same arithmetic in the same order as `kvshift.rerotate_keys`, so the outputs agree bit-for-bit almost everywhere; `tests/test_shift_kernels.py` (GPU-only) checks δ ∈ {−3000, −4, 0, 4, 186, 10000} × {HND, NHD} × {aligned, ragged} against the torch path with a scattered block table. Both paged layouts are handled by passing strides rather than branching, and a torch fallback is selected when Triton or CUDA is unavailable (`MARATHON_NO_TRITON` forces it).

Micro-benchmark, Qwen3-14B shapes (40 layers × 8 KV heads × 128 dim, bf16, δ=186, best of 5; GB/s counts the source read plus the destination write, which is the memcpy floor for the op):

```
tokens    MB | torch ms   GB/s  us/tok | triton ms   GB/s  us/tok | speedup
  4096   640 |     6.52  191.6   1.593 |      1.01 1233.5   0.247 |   6.4x
 12288  1920 |    20.96  178.9   1.706 |      2.80 1339.3   0.228 |   7.5x
 30720  4800 |    61.48  152.5   2.001 |      6.54 1433.0   0.213 |   9.4x
```

1,433 GB/s at 30k tokens is ~95% of the 5090's ~1.5 TB/s device-to-device peak — the copy is now memcpy, and at 0.213 µs/token it beats the 0.35 µs/token target by 1.6× and the old in-connector rate by ~23×. A replication at real connector sizes (40 layers of 56k-token paged cache, 5.4 GB store, scattered block table) gives the same 6.2 ms / 1,433 GB/s, so the number is not an artefact of small tensors.

30k length-sweep point re-measured, prefix vs shift, same recipe as the north-star entry (`--turns 54 --edit-at 50 --parity-tokens 16 --gpu-util 0.78 --store-tokens 33280`):

```
                        before (torch copy)   after (Triton copy)
edit turn, shift             0.3663 s              0.2603 s
  of which copy_ms            151.6 ms              34.4 ms
edit turn, prefix            3.8688 s              3.9391 s
speedup                        10.6x                 15.1x
steady-state prefill    0.156/0.162 s         0.1583/0.1633 s  (prefix/shift)
```

Copy is **4.4× cheaper** and now 13% of the edit turn instead of 41%; the edit turn itself is **1.41× faster**, and the gap over prefix caching at 30k widens from 10.6× to 15.1×. All 54 turns' generated text is identical between the two modes and the parity answer is still `7391-KAPPA`. Steady-state prefill is untouched, as it must be — nothing changes on unchanged turns.

**Two traps worth recording, because both would have been misread as "the kernel is slow".** The first in-engine measurement came back at 216 ms — *worse* than torch. An edit turn issues exactly one load, so Triton's JIT compile was being charged in full to the one thing it was meant to accelerate. Warming the kernel at `register_kv_caches` time fixed it — but only to 74 ms, because Triton specialises on argument divisibility and a 1-token warmup compiles a different kernel from the 29,232-token load that follows. Warming with both a 16-divisible and a ragged token count got the real number, 34.4 ms.

**What the remaining 34.4 ms is.** The kernel itself moves those 9.1 GB in 6.2 ms. The rest is one-shot cost around it — 40 store reads, the slot transfer, and first-touch of buffers written by the preceding turn's saves — paid once per edit turn, not per token, and the isolated replication shows the same cold/warm split (9.8 ms first pass, 6.2 ms after). Chasing it is worth ~28 ms of a 260 ms turn and is not done. What *is* now true: the copy is no longer the slope. At 30k the shift edit turn is 0.26 s against prefix caching's 3.94 s, and 87% of it is vLLM prefilling the edited span and the new tokens — the term the design says has to be there.

Caveats: one point of the sweep (30k), not the whole curve, so the new slope in context length is not measured — only its largest point. Single runs, and the earlier 74 ms run also showed several steady turns spiking to 2–4 s under contention from another agent's GPU job, which is a reminder that any single cell here is one sample. Parity is still greedy-text equality against prefix mode plus one planted fact. The kernel is bf16-only in practice (it inherits whatever dtype the cache has, but only bf16 has been run), single-GPU, and untested under tensor parallelism like everything else in the connector.

@acrosley 2026-08-19

## 2026-08-19 — The pieces become a system: `marathon.server` + `marathon.client` end to end, 12.6k-token edit turn at 0.179 s vs 1.277 s over HTTP

Everything measured so far was a probe driving the engine directly. This entry is the first end-to-end run of the actual thing: a client that holds a conversation and ships only deltas, an HTTP endpoint that verifies each payload against a content-addressed store, and a server that plans KV reuse from the *verified* state and drives vLLM with the shift connector. `src/marathon/server.py`, `src/marathon/client.py`, `scripts/server_demo.sh`. The phase driver is no longer duplicated — `local_probe._phases` moved to `reuse_plan.phases` and both callers use it.

**Qwen/Qwen3-0.6B, 12 turns, edit at turn 9, over `POST /v1/turn`** (`gpu_memory_utilization=0.30`):

```
turn  wire_bytes  state_bytes  prompt_tokens  prefill_s  reused  phases  policy   reply
   0        7122         2987            600     0.2327       0       1  first    'Reply: Ok.'
   1        3398         6018           1202     0.0185       0       1  reuse    'Reply: Ok.'
   4        3321        15142           3021     0.0195       0       1  reuse    'Reply: Ok.'
   8        3321        27266           5429     0.0213       0       1  reuse    'Reply: Ok.'
   9        3519        30375           6049     0.0322    4829       2  reuse    'Reply: Ok.'   <- edit turn
  10        3397        33407           6652     0.0247       0       1  reuse    'Reply: Ok.'
  11        3495        36481           7264     0.0303       0       1  reuse    '7391-KAPPA'   <- planted fact
```

**Qwen/Qwen3-14B-FP8, 24 turns, edit at turn 20, with the reuse control (`--no-reuse`, plain vLLM prefix caching) run separately:**

```
turn  prompt_tokens  shift prefill_s  control prefill_s  reused tokens  phases
   0            600          0.1778             0.1793              0       1
  10           6604          0.0907             0.0912              0       1
  19          12004          0.1090             0.1078              0       1
  20          12622          0.1793             1.2770          11404       2   <- edit turn, 7.1x
  21          13222          0.1230             0.1175              0       1
  23          14431          0.2020             0.1957              0       1   <- parity: '7391-KAPPA' both
```

**The headline is the edit turn: 1.277 s → 0.179 s, 7.1×, through the full protocol path** — payload verification, delta reconstruction, chat-template rendering, plan, k+1 connector requests and generation, not a probe shortcut. It lands inside the 0.16–0.25 s band the direct probe measured at this length, so the protocol layer costs nothing detectable. Every non-edit turn is within noise of the control, which is what it must be: nothing changes on an unchanged turn. Both runs answer the planted fact `7391-KAPPA` on the last turn, so the reuse did not lose the fact it stitched over.

**The wire column is the other half of the claim, and it is the one only an end-to-end run can show.** State grows from 3 KB to 72 KB across 24 turns; the payload stays at ~3.3 KB every single turn, including the edit turn (3529 bytes — a rewritten opening message costs 200 bytes over an append). Per-turn wire cost is flat while the conversation grows linearly, which is the DESIGN.md claim stated as bytes rather than as prefill.

**A real bug the end-to-end path found that no probe could have.** The reuse plan needs the token ids *each message contributes*, and the obvious way to get them from a chat template — render `messages[:k+1]`, subtract `messages[:k]` — is wrong, because chat templates are not append-only. Qwen3 renders a *trailing* assistant message with an empty `<think>` block and silently drops it once another message follows. Every coordinate after the first assistant turn would have been shifted by four tokens, and the connector would have transplanted KV into the wrong slots — a plausible-looking prompt with quietly corrupted reuse. `ChatTokenizer` now renders each prefix with a throwaway sentinel message appended, so no real message is ever last; the sentinel's rendered block is derived from the template rather than hardcoded, the prefix property is checked per message rather than trusted, and the ids fed to the engine are always a single encode of the full prompt with the pieces supplying only lengths. The probes never hit this because they built their own `role: content` prompt layout and never went through a chat template at all.

**Session handling.** Previous state, per-line token cache and connector store key are all keyed by session id, and CPU tests (`tests/test_server.py`, fake engine, no vLLM) cover the three properties that matter: a tampered delta or a bad `target_hash` raises before the engine is ever called, two interleaved sessions plan independently (an edit in one leaves the other append-only), and an append-only turn asks the KV layer for nothing — one segment, zero reused tokens handed to the connector, one request, because the leading prefix is vLLM's own cache and not ours.

**Honest limits.** The connector store key carries an epoch that rolls after an edit turn, so a session's *second* edit re-saves under a fresh key with partial coverage and the store declines what it does not hold — safe (a recompute, never a wrong answer) but not fast. v1 therefore accelerates the first edit in a session and degrades gracefully after it; making repeated edits fast needs a save path that can write the loaded segments' new positions, which is not built. The demo is one run per configuration, not a distribution. The client advances its baseline when it builds a payload, so a rejected turn leaves it ahead of the server and the documented recovery is to drop the session. Single GPU, no tensor parallelism, and the HTTP layer is stdlib `http.server` with a lock around a blocking single-tenant engine — correct for one conversation at a time, not a serving front end.

@acrosley 2026-08-19
