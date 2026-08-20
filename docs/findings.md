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

## 2026-08-19 — Phase 1.5: on a Gated-DeltaNet hybrid the trick survives, but only if the linear layers get their own cache — and one new failure mode appears

Commands: `scripts/kvshift_hybrid.sh --turns 20 --max-new-tokens 12 --first-m 256` and the same with `--scenario edit-turn0` (which adds `replay-mix+first256`) · Model: `Qwen/Qwen3.5-4B` (bf16, HF transformers 5.15, sdpa, RTX 5090 / WSL2) · Code: `src/marathon/kvshift_hybrid.py`, `src/marathon/kvshift_hybrid_probe.py` · Cost: $0 · logs `~/marathon-logs/kvshifthyb_*.log`

PLAN.md parked hybrid degradation as a Phase 1.5 question. This is the answer, on the model an earlier entry rejected for exactly this reason. `Qwen/Qwen3.5-4B` confirmed from `config.json`: 32 layers, `full_attention_interval=4` → **24 linear (Gated DeltaNet) / 8 full attention**, `head_dim=256` with `partial_rotary_factor=0.25` (only 64 dims rotate) and interleaved mRoPE. The mRoPE turns out to be a no-op for text — all three position rows are equal, so the interleaved sections carry identical frequencies — and re-rotation still holds exactly: **max abs error 8e-6 – 1.5e-5** for δ=37, measured through `model.model.rotary_emb` itself rather than a reimplementation of it. `rerotate_keys_partial` rotates the leading 64 dims and leaves the rest alone; that is the entire delta to `kvshift.py` on the attention side. `flash-linear-attention` is **not** installed, so the recurrence runs transformers' pure-torch fp32 chunked scan — which matters for the wall-clock column and nothing else.

**The structural problem.** A linear layer has no per-token KV, so there is nothing to re-rotate and no position to re-rotate it to. Its state is a running summary `[32 heads, 128, 128]` plus a 4-wide conv window. "Move S by δ" can only mean "run S's tokens through the recurrence again". Four policies (all in `kvshift_hybrid.py`; the 8 attention layers get the unchanged kvshift treatment under every one of them):

- **stale-state** — linear layers reuse the state cached at the end of the *old* context and append E' to it. The naive serving baseline: E' reaches the query through the 8 attention layers, and in the linear half only as 21 tokens tacked on after S.
- **replay-hidden** — cache, from the old turn, each linear layer's per-token *input* hidden states over S, plus the state at the end of P. On the edit: run E' fresh through all layers, then per linear layer roll only its recurrence over S from those stale inputs. Stale per-token inputs, fresh aggregation — the same staleness class as re-rotated KV. Skips the MLP, which is 63% of a linear layer's weights.
- **replay-mix** — the same idea one level deeper: cache the old turn's post-conv `(q, k, v, beta, g)` for S. That *is* the linear layer's KV cache, and replay collapses to the bare scan.
- **+first{M}** — extend the fresh chunk M tokens past E' into S.

Measured per-token cost (parameter counts read off the loaded model): a full token-forward is **3,620 M**; one replayed S token across all 24 linear layers is **558 M for replay-hidden (15.4% of a token-forward)** and **50 M for replay-mix (1.39%)**. Those two numbers are most of the story: 15.4% is a floor you cannot get under while re-projecting the hidden states, and 1.39% is what caching the mixer's own q/k/v buys instead.

Below, `kl fact` is the worst klmean over the three planted-fact questions (prefix fact in P, edit fact in E', suffix fact in S), `kl open` is klmean on the open-ended summary, `facts` counts exact answers, `flops` is the fraction of a full recompute's weight FLOPs, and `prefill_s` includes the query.

```
edit-turn0   P=41  E=17->21 (d=+4)  S=5271  N=5333
  policy                  flops  prefill_s  vs full  kl fact   kl open  tf_top1(open)  facts
  full-recompute          1.000     0.675     1.0x    0.0000    0.0000      1.00        3/3
  no-rerotate (control)   0.008     0.200     3.4x    0.0083    0.0724      0.88        3/3
  stale-state             0.008     0.189     3.6x    0.0154    0.0755      0.88        3/3
  replay-mix              0.022     0.574     1.2x    0.0395    0.0228      0.92        2/3  <- misses the edit
  replay-mix+first256     0.069     0.588     1.1x    0.0141    0.0166      0.90        3/3
  replay-hidden           0.160     0.618     1.1x    0.0343    0.0211      0.92        2/3  <- misses the edit
  replay-hidden+first256  0.200     0.633     1.1x    0.0037    0.0143      0.96        3/3

edit-mid     P=2642  E=18->22 (d=+4)  S=2669  N=5333
  full-recompute          1.000     0.664     1.0x    0.0000    0.0000      1.00        3/3
  no-rerotate (control)   0.008     0.198     3.4x    0.0144    0.0437      0.88        3/3
  stale-state             0.008     0.204     3.3x    0.0150    0.0451      0.90        3/3
  replay-mix              0.015     0.420     1.6x    0.0017    0.0139      0.92        3/3
  replay-hidden           0.085     0.476     1.4x    0.0016    0.0103      0.92        3/3
  replay-hidden+first256  0.126     0.491     1.4x    0.0005    0.0071      0.92        3/3

edit-grow    P=2642  E=257->958 (d=+701)  S=2430  N=6030
  full-recompute          1.000     0.798     1.0x    0.0000    0.0000      1.00        3/3
  no-rerotate (control)   0.162     0.303     2.6x    0.0231    0.0461      0.96        3/3
  stale-state             0.162     0.312     2.6x    0.0175    0.0296      0.96        3/3
  replay-mix              0.168     0.511     1.6x    0.0020    0.0101      0.96        3/3
  replay-hidden           0.224     0.566     1.4x    0.0012    0.0060      1.00        3/3
  replay-hidden+first256  0.260     0.570     1.4x    0.0026    0.0049      1.00        3/3
```

**What survives.** On a mid-history edit, `replay-mix` reaches klmean **0.0017** on facts and **0.0139** open-ended at **1.5% of full-model FLOPs**. The dense Qwen3-8B distribution eval's headline was 1.5–1.6% of tokens forwarded for a median KL of 0.0035 — so on the compute axis essentially *all* of the dense win transfers, at comparable or better KL. Re-rotation is still doing real work on the 8 attention layers: the `no-rerotate` control is worse than `stale-state` at identical cost on the scenario where δ is large (edit-grow, δ=+701: 0.0231 vs 0.0175 on facts, 0.0461 vs 0.0296 open), and roughly ties it where δ=+4, which is what a 4-token shift should look like.

**What it costs that a dense model does not.** Memory. The 8 attention layers' KV for 5.3k tokens is **167 MiB (32 KB/token)**. The linear layers' mix cache over the same span is **2,988 MiB (566 KB/token)**, and the hidden-state cache **618 MiB (117 KB/token)**. Buying back the dense FLOP fraction on a hybrid costs roughly **18x the cache bytes per token**. If you will not pay that, `replay-hidden` is 5x cheaper in memory and 6x more expensive in FLOPs (8.5–22%), and `stale-state` is nearly free (0.8%) but 5–25x worse in KL.

**The new failure mode, which dense models do not have.** On `edit-turn0` — the edit sits 41 tokens in, with 5,271 tokens of S after it — both replay policies answered the *pre-edit* code `7391-KAPPA` when asked for the edited fact, while `stale-state` got it right. That inversion is not noise, it is the recurrence decaying: replay rolls E' into the state and then washes it out under 5,271 replayed tokens, whereas `stale-state` appends E' *last*, so recency saves it by accident. It is the mirror image of the dense finding, where the failure class was *governing* spans and `--repair-first` did not help. Here first-M is exactly the repair that works, and it is cheap: `replay-mix+first256` restores the fact at 6.9% of FLOPs (klmean on that question 0.0395 → 0.0141), `replay-hidden+first256` at 20% (0.0343 → 0.0037). On `edit-mid` and `edit-grow`, where S is half as long, no repair is needed at all. So the rule this points at — **recompute the first M tokens of S when |S| after the edit is large relative to the state's effective memory** — is about the ratio, and one scenario is not enough to calibrate M.

**Wall clock is not the FLOP number, and the reason is not the method.** `replay-mix` is only 1.2–1.6x faster than full recompute despite doing 1.5–2.2% of the weight FLOPs, because the replay runs transformers' `torch_chunk_gated_delta_rule`: a Python loop over 64-token chunks in fp32, ~40 chunks x 24 layers per edit turn, with no fused kernel behind it. `stale-state`, which does no scan at all, gets the 3.4x its FLOP count predicts. The honest reading is that the FLOP column is the method's result and the wall column is this prototype's — the same gap the dense work closed by moving `kvshift` into vLLM and then onto a Triton kernel.

**Verdict.** Delta-driven reuse does survive on a GDN hybrid, but it stops being a *cache* trick and becomes half cache, half recompute: 8 layers reuse KV exactly (re-rotation is still exact under partial rotary and mRoPE), and 24 layers must replay their recurrence over S. With the linear layers' own q/k/v cached, the edit turn costs 1.5–2.2% of full-recompute FLOPs at KL comparable to the dense result — the compute win transfers — but it needs ~18x the cache memory per token, needs a fused scan kernel before the wall clock follows, and needs first-M repair for edits far from the end of a long context. `stale-state`, which is what a serving system would do without thinking about it, is 5–25x worse in KL and visibly worse in free-running text (tf_top1 0.88–0.90 against 0.92–1.00 on the summaries), so "just keep the final state" is not good enough.

Caveats: one model, three scenarios, one session shape, single runs — no distribution eval like the dense 144-session one, so every cell here is one sample. HF eager prototype, not a serving engine. Facts are graded on a 12-token forced continuation and summaries on 48 greedy tokens. δ=+701 on `edit-grow` because the builder's `grow` argument is characters, not tokens. The old-turn capture prefills in two chunks (P, then E+S), so its state differs from a one-shot prefill in the last floating-point bits. Only `edit-turn0` got the `replay-mix+first256` cell.

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

## 2026-08-19 — Every edit in a session now costs the same as the first: the store is rebuilt in the new coordinates for +5 ms

The previous entry shipped `marathon.server` with an honest hole: only a session's *first* edit was fast. The connector's store key carried an epoch that rolled after an edit turn, so the second edit planned against a layout the first had replaced, the store declined the load, and the turn fell back to a full recompute. This closes it.

**Why it broke.** `ShiftStore` is a flat position-indexed buffer — index *is* the token's absolute position. That is only meaningful in one set of coordinates at a time. An edit changes the length of the edited span, so everything after it moves; the reused span's KV is still correct, but its *index* now names the wrong position, and the freshly computed span cannot be written where it belongs without colliding with the old layout. No amount of bookkeeping on top of a contiguous buffer fixes that: the two layouts genuinely disagree about where things live.

**The fix is one word in `kv_transfer_params`.** The edit turn's final request now sends `"save": "full"`, and the connector's `_plan_save` widens the save from "the positions this step computed" to `[0, hi)`. By the time `save_kv_layer` runs, the whole prompt is resident in the paged KV cache — the loaded span was scattered in by `start_load_kv`, the prefix hits are in shared blocks, the rest was just computed — so re-gathering it writes the store back at the *new* positions. `reserve` already treats a save at position 0 as a truncating rewrite, so the store comes back contiguous and the session is append-only again. The next edit is then an ordinary first edit. All of a step's loads are issued in `start_load_kv` before any `save_kv_layer`, so the re-read cannot race the load that fed it. The server's epoch bookkeeping is deleted.

**Qwen3-0.6B, 24 turns, edits at 8 / 14 / 20, a code planted three turns before each:**

```
edit turn   prefill_s   tokens reused   connector load
        8      0.0397            4227   store[604:4812]   delta=20  copy 2.33 ms
       14      0.0451            7856   store[620:8460]   delta=20  copy 1.57 ms
       20      0.0527           11487   store[636:12108]  delta=20  copy 2.03 ms
```

Answer on turn 23: `7391-KAPPA, 5820-OMEGA, 1146-SIGMA` — all three, including the code that has now survived three separate edit turns. Zero `declining reuse`, `refused save` or `no stored KV` warnings in the run. The load windows are the proof the fix works: edit 2 reads `store[620:8460]`, a window that only exists if the store was rewritten in the post-edit-1 coordinates. Under the old epoch scheme that read would have been declined.

**Qwen3-14B-FP8, 24 turns, edits at 12 and 20, against the `--no-reuse` control:**

| edit turn | history | control | Marathon shift | speedup | tokens reused |
|---:|---:|---:|---:|---:|---:|
| 12 | 7.8k | 0.651 s | **0.160 s** | 4.1× | 6604 |
| 20 | 12.6k | 1.287 s | **0.184 s** | 7.0× | 11417 |

The second edit is as fast as the first was, and matches the previous entry's single-edit number at the same length (0.179 s at 12.6k) to within 5 ms. Both modes answer `7391-KAPPA,5820-OMEGA`.

**What the full re-save costs: about 5 ms, far less than expected.** The previous entry's single edit at 12.6k took 0.179 s with no save; the same edit with the whole 12.6k-token prompt re-gathered across 40 layers takes 0.184 s. The gather rides on memory bandwidth that the turn is not otherwise using, and it is paid only on edit turns — steady-state turns are untouched (0.09–0.12 s, identical to the control). Cheap enough that no cleverer scheme is worth building: the alternative designs (a free-list store with a logical→physical interval map, or a server-side remap mirroring the store) both trade this 5 ms for a substantially more complex invariant, and the store's contiguity is what makes the scheduler/worker mirror agree in the first place.

**Limits.** Chunked prefill would make a `"full"` save re-gather a growing prefix on every step — correct but wasteful; chunked prefill interleaved with a load was already documented as untested and the server does not produce it. The re-save assumes the whole prompt is resident in the request's blocks, which is true for the phase driver's final request and not checked. Two runs per configuration, one length sweep point each. Everything else from the previous entry still stands: single GPU, no tensor parallelism, stdlib HTTP with a lock around a blocking engine.

@acrosley 2026-08-19

## 2026-08-19 — Phase 3 reframed, and a stitched-KV fine-tuning pilot that is honestly negative: at 0.6B the failure it repairs isn't there

Command: `scripts/stitch_train.sh --items 600 --eval-items 120 --min-tokens 3000 --max-tokens 5000 --gen-tokens 32 --tag pilot` (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, eager, bf16, RTX 5090; train 599 items in 937 s, eval 120 held-out items; LoRA r=16 alpha=32 on q/k/v/o, AdamW 1e-4, accum 4, anchor weight 1.0 every other step) · Model: `Qwen/Qwen3-0.6B` · Cost: $0.

**First, the reframe** (written up in [phase3-design.md](phase3-design.md); PLAN.md's Phase 3 section rewritten). DESIGN.md's Phase 3 was "fine-tune the model to consume delta-formatted *text* and trust that absence-from-diff means unchanged". Phase 1 already got that outcome without training, and more completely: position-shifted KV reuse never retransmits, re-tokenizes or re-prefills the baseline, yet the model's attention still runs over the full history, so its input distribution never changes and there is nothing to persuade it of — 1.5–1.6% of tokens forwarded for median KL 0.003, planted-fact retrieval 105/111 against full recompute's 106/111. That bet is superseded. What remains is the harder form: the one residual failure class (an edit to a *governing* span leaves the reused suffix conditioned on an instruction that is gone — ~9x KL, all of the >0.2 tail, and selective recompute does not fix it), and more generally making the model robust to stitched caches so reuse can be pushed into cases the planner currently refuses (relocated blocks, large delta, the hybrid tier's washed-out state). The method under test: **stitched-KV consistency fine-tuning** — LoRA, forward passes run *with* the stitched cache (P verbatim, E' fresh, S re-rotated), loss = KL to the frozen base model's full-recompute continuation, plus an anchor term running the student on a *clean* cache against the same teacher.

Two properties make it the honest version rather than a proxy. The training objective **is** the eval metric — mean KL over 32 teacher-forced continuation tokens against full recompute is literally the loss, so there is no gap between "the loss went down" and "the reported number went down". And the anchor's floor is a true zero: teacher and anchor share one function, so at identity they are bit-identical. That took a fix. The first launch reported `clean_kl = 0.0049` at step 0, where an identity adapter must give 0 — prefilling the anchor separately from the teacher had put a ~0.005 bf16 floor under the damage metric, *larger than the damage it has to resolve*. After the fix it reads 0.0000, and a test pins it.

**The pilot, on 120 held-out sessions (seed 9001, never trained on; training was seed 7001):**

```
bucket          metric               n      mean    median       p95       max  >.05
governing       base_stitch_kl      57    0.0034    0.0024    0.0063    0.0207     0
governing       tuned_stitch_kl     57    0.0034    0.0028    0.0069    0.0118     0
governing       tuned_clean_kl      57    0.0033    0.0027    0.0085    0.0123     0
non-governing   base_stitch_kl      63    0.0023    0.0018    0.0055    0.0088     0
non-governing   tuned_stitch_kl     63    0.0032    0.0026    0.0069    0.0126     0
non-governing   tuned_clean_kl      63    0.0029    0.0024    0.0055    0.0101     0
ALL             base_stitch_kl     120    0.0028    0.0022    0.0060    0.0207     0
ALL             tuned_stitch_kl    120    0.0033    0.0027    0.0069    0.0126     0
ALL             tuned_clean_kl     120    0.0031    0.0025    0.0062    0.0123     0

planted-fact ok   governing 57/57 for ref = base = tuned;  non-governing 60/63 for ref = base = tuned
base   governing/non-governing KL ratio:  mean 1.50x   median 1.35x
tuned  governing/non-governing KL ratio:  mean 1.05x   median 1.10x
```

**The headline is the base column, not the tuned one: the failure class did not reproduce.** On this population a governing edit costs mean KL 0.0034 against non-governing's 0.0023 — a **1.50x** ratio, where the 144-session 8B eval measured ~9x. The worst single item in 120 is 0.0207 and **zero** items clear KL 0.05, against that run's 7/78 governing items over 0.05 and 2 over 0.2. The dependent-edit probe agrees:

```
scenario          question           base_kl  tuned_kl  clean_kl  ref_ok base_ok tuned_ok
dep-instruction   lang-pipeline       0.0038    0.0029    0.0027    True    True     True
dep-instruction   lang-scheduler      0.0078    0.0071    0.0099    True    True     True
dep-anaphora      primary-key         0.0053    0.0062    0.0020    True    True     True
dep-anaphora      mission             0.0032    0.0023    0.0024    True    True     True
dep-anaphora      open                0.0040    0.0121    0.0070    None    None     None
dep-contradict    harbor              0.0049    0.0105    0.0033    True    True     True
dep-contradict    open                0.0038    0.0040    0.0027    None    None     None
```

`dep-instruction` — the scenario that broke at 8B with first-token KL 0.3492 and 0/2 agreement — here scores klmean 0.0038 and 0.0078 and matches the reference on both questions. There is nothing to repair.

So the adapter was trained against a population with almost no signal to exploit, and it behaves accordingly: **it did not improve stitched KL** (mean 0.0028 → 0.0033, marginally *worse*; the governing/non-governing ratio "improves" from 1.50x to 1.05x only because the non-governing cases got worse, not because governing got better), and it spent **0.0031** of clean-context drift buying that. Against the exit criteria in phase3-design.md: criterion 1 is untestable on this data; criterion 2 (clean-context KL ≤ 0.002) **fails**; criterion 3 (no regression on what already works) **fails**. Planted-fact accuracy is the one thing untouched — 117/120 for reference, base and tuned alike — so nothing was broken outright, but the two probe scenarios that were already fine got worse under the adapter (`dep-anaphora/open` 0.0040 → 0.0121, `dep-contradict/harbor` 0.0049 → 0.0105).

The training curve says the same thing from the other side: governing `stitch_kl` median 0.0053 → 0.0031 across the run, non-governing 0.0059 → 0.0032, and `clean_kl` 0.0048 → 0.0028 — all three fall *together*, which is what converging back toward the identity adapter looks like after an early over-large-LR transient (`clean_kl` peaked at 0.0223 near step 81). The optimiser found nothing better to do than undo itself.

**Why the population is wrong, and it is three things at once.** (i) **Model**: 0.6B, where the 9x was measured at 8B — the 2026-08-18 probe entry already recorded the 0.6B model's weaker instruction-following, and a model that barely obeys a standing instruction cannot be much disturbed by carrying the stale version of it in the suffix. (ii) **Length**: 3–5k-token sessions against the eval's 4–8k, chosen for iteration speed. (iii) **Query**: `build_examples` takes `item.queries[:1]` and `kvshift_eval` always puts `fact-at` first, so all 120 eval items asked the same question type and the `obey` query — the one a governing edit is most directly aimed at — never ran. That third one is a plain defect in the harness, not a judgement call.

**Verdict: negative, and the negative is about the testbed rather than the method.** What this run does establish is that the instrument works end to end and is trustworthy: the differentiable stitch places tensors identical to the serving path's, gradients reach the adapter through a stitched cache, the split backward is arithmetically identical to the summed one, and the anchor reads exactly 0 at identity so damage is measurable below the eval's own ~0.0015 numerical floor — all pinned by CPU tests on a random tiny Qwen3. What it does **not** establish is anything about whether stitched-KV consistency fine-tuning fixes governing edits, because the governing edits in this pilot were not broken. phase3-design.md called this the smallest honest experiment; it was too small on the one axis that carries the phenomenon.

The next run is specified and was not made: **Qwen3-8B, 4–8k sessions, the full query pool including `obey`, and a base-only eval first** to confirm the ~9x reproduces *before* spending an epoch of training on it. That confirmation was launched and immediately stood down — Track N's Phase 2 cold-tier eval had taken the GPU back (32.1 of 32.6 GB: `marathon.cold_eval` plus a vLLM engine), and the etiquette here is to yield, not to queue behind it.

**One infrastructure lesson, learned the expensive way.** The first attempt died at item ~210 with `torch.AcceleratorError: CUDA error: unknown error` after degrading from 1.9 s to 19 s per item. Nothing else was on the GPU; it was self-inflicted allocator thrash — `torch.cuda.empty_cache()` on *every* step (each call hands every cached block back to the driver, so the next step re-`cudaMalloc`s ~2 GB of KV) on top of holding the stitched and anchor autograd graphs at the same time. The two loss terms are independent, so they are now backpropagated separately and freed as they go (halving peak memory, with a test asserting the gradients match the summed form), `empty_cache` is every 25 steps, and the script exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the varying 3–6k buffer sizes. The rerun held a flat 1.6 s/item for all 599 items. At 8B this matters more, not less.

Caveats: one model, one seed pair, one epoch, one query type, 120 held-out items. `gov_frac 0.5` gives ~282 governing and ~317 other training items, near the intended split but not tuned. No hyperparameter search was run — 1e-4 was a first guess and the `clean_kl` spike by step 81 suggests it is too high for r=16 across all four projections. The eval's `*_answer_ok` grades the arg-max of the teacher-forced sequence rather than a free-running greedy decode; that is comparable across base and tuned but is not `kvshift_eval`'s `exact` column. And the method distills toward full recompute, which the dependent-edit study showed is *itself* sometimes the less obedient party — so a KL improvement would not have been an accuracy improvement even had one appeared.

## 2026-08-19 — Phase 2 cold tier: the window goes flat, and recall-on-miss is the whole difference between 0.008 and 0.817

Command: `python -m marathon.cold_eval --sessions 20 --turns 70 --active-window 8192 --threshold 0.2` (`scripts/cold_eval.sh`) · Model: `Qwen/Qwen3-14B-FP8` in vLLM 0.27.1, retriever `sentence-transformers/all-MiniLM-L6-v2` mean-pooled on CPU · 20 sessions × 70 turns, history p50 25.0k / min 18.5k / max 35.7k tokens, 6 facts planted per session from turn 2 to turn 65, 8 questions each (old / recent / distractor) = 480 questions.

`marathon.cold` keeps the model-facing view under a token budget by demoting the oldest non-governing messages to a stub carrying the message's own content address, `[cold #12 3f9a1c04: <first ~12 words>]`. The full bytes stay in the ledger, so a demotion is a *shrink edit of the view* and a promotion a *grow edit* — the shape `reuse_plan` already handles. Recall-on-miss has two triggers: exact (the turn's delta touches a demoted message) and query (top-k≤2 chunk-embedding matches above a cosine threshold).

All three conditions run with the shift connector **off**, on plain vLLM prefix caching, so they differ only in the paging. Conditions share one engine and one frozen corpus.

**The three exit criteria.**

| condition | active p50 | active max | em old | em recent | em all | promo recall | promo precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| full (reference) | 13078 | 35674 | 0.900 | 0.983 | **0.942** | — | — |
| cold-norecall | 7647 | 8062 | 0.000 | 0.017 | **0.008** | 0.00 | — |
| cold-recall | 7681 | 8075 | 0.850 | 0.783 | **0.817** | **0.912** | 0.291 |

**1. Bounded window on unbounded sessions — met.** Median active tokens by turn index, over the 20 sessions:

```
condition        turn10   turn30   turn50   turn70      max
full               3755    11203    18352    25691    35674
cold-norecall      3755     7830     7713     7640     8062
cold-recall        3755     7916     7760     7655     8075
```

The reference grows linearly and the paged conditions are *flat from turn 30 on* — 7916 → 7655 as the history doubles, never exceeding the 8192 budget by more than one message (8075). At the end of a session 121 messages are cold, at 3.19 demotions per turn. Flatness needs one thing beyond stubs: a stub is ~20× smaller than the message it replaces, but 20× smaller is still O(n), so once nothing is left to demote the oldest stubs are **evicted** from the view entirely. Nothing is lost — the bytes are in the ledger and the retriever searches every demoted message, evicted or not — but this is the step that makes the window bounded rather than merely smaller, and CPU tests assert it directly over 120-turn synthetic sessions at four budgets.

**2. Recall-on-miss restores demoted content — met.** 102 of the 120 fact questions had their answer in a message that was cold when the question arrived. The query trigger promoted the right message for **91.2%** of them, and that is the whole difference in the table: the same paging policy scores 0.008 without recall and 0.817 with it. Stubs alone are worthless for answering — which is the honest reading of `cold-norecall`, and the reason the design doc calls a wrong demotion a silent degradation of ground truth.

Retrieval had to be scored per *chunk*, not per message. Pooling one vector over a 400–700 token turn averages away the single sentence that answers the question, and MiniLM truncates at 512 tokens anyway, so a fact late in a long turn is invisible: whole-message pooling recalled the right message **39%** of the time on this same question set. Scoring a message by its best 60-word overlapping window took it to 91%. Precision is 0.291 against a ceiling of 0.5 — `top_k=2` fires two promotions per question and only one can be the target — so the wasted promotion is a deliberate cost, not a miss.

**3. Quality delta vs full-context replay — partially met, and I am not going to call it a pass.** 0.817 vs 0.942 is **−12.5 points**, recovering 87% of the reference. That is a large improvement over the naive baseline and still a real regression against full context; "efficiency that changes answers is a regression" is the rule, and a 12.5-point drop is not within any tolerance worth the name. Where it goes: 8.8% of fact questions have their answer cold *and* unpromoted, and the residue is the model failing to use a message that was correctly promoted. Note `em_recent` (0.783) is *below* `em_old` (0.850) — with `keep_last=6` on a 70-turn session even "recent" facts are cold by question time, so that column is not the in-window control it was meant to be. No condition fabricated a code on a distractor (0.000 everywhere).

**The exact trigger is not exercised by this eval.** It never fires here, because the eval only appends and questions, never edits history. It is covered only by CPU tests (delta touches a demoted message → promoted regardless of similarity). A GPU eval of edit-into-cold-content is still owed.

**Cost: paging makes per-turn prefill worse, not better, and the connector does not yet rescue it.** Demoting the oldest message edits the *front* of the view on every single turn, so prefix caching cannot help on any turn and each turn is a full recompute of the (smaller) window. With every history turn actually prefilled (`--generate-history`):

| condition | prompt | prefill p50 | p90 | max |
|---|---:|---:|---:|---:|
| full, prefix caching | up to 35.7k | **0.083 s** | 0.149 | 0.23 |
| cold, prefix caching | 8k | **0.586 s** | 18.6 | 38.3 |
| cold, **shift connector on** | 8k | **6.18 s** | 12.1 | **382** |

A 4.4× smaller window costs 7× more prefill per turn, because the reference is append-only and the paged one is not. Shifted KV reuse is precisely the mechanism that should fix this — a demotion is a shrink edit, a promotion a grow edit — and turning it on today makes it an order of magnitude worse still, with a 382 s outlier. Worse, it is not only slow: in a pilot on the same policy and sessions, connector-off scored **1.0** exact-match and connector-on **0.33**, degrading progressively across successive turns (the first three questions right, then everything after wrong). Phase 1 measured *one* edit per session; a paged session is 70 consecutive edit turns, and repeated per-turn edits through the connector are not yet correct at that depth even with the `save="full"` rebuild. **This is the Phase 2 / Phase 1 composition failing, and it is the thing to fix before the cold tier is worth serving.** Until then the cold tier is a correctness/capacity feature, not a latency one.

**Two sizing traps, both of which cost hours.** The shift store is carved out of the GPU budget *before* vLLM sizes its KV cache: `store_tokens=65536` on Qwen3-14B is 10.7 GB, and at `gpu_memory_utilization=0.85` that yields `Available KV cache memory: -2.47 GiB` and a refusal to start — just under the threshold it yields a near-zero KV cache, 15–37 s prefills and preemption. A 24k store at 0.93 gives 12.66 GiB / 82,928 tokens and is what the run above uses. Separately, the retriever must not share the card: MiniLM allocating into the ~7% vLLM leaves crashes the run with `cudaErrorUnknown` mid-session, so `TransformerEmbedder` defaults to CPU, which costs nothing at 22M parameters.

**One eval bug worth recording, because it inverted a result.** The first run of this eval planted each fact in the *opening words* of its message. The stub keeps the opening words. So the no-recall baseline could read the answers straight off the stubs and scored 0.5 on `em_old` — the naive baseline looked far better than it is. Facts now go after the body, and `tests/test_cold_eval.py` asserts that no planted fact is recoverable from any stub or from the paged view, over six sessions.

**Limits.** One model, one window size (8192), one threshold (0.2), one `keep_last` (6); no sweep of any of them. The `full` reference comes from a separate invocation of the same seeded sessions and frozen corpus (the eval now reads `tests/data/kvshift_eval_corpus.json`, not the working tree, precisely so that runs of different conditions are comparable — building sessions from repo source meant that editing these docs moved every session). History turns in the headline run are not prefilled (`generate=False` — verify, page, render and plan all still run), which is what makes the eval affordable; the state each question is asked against is identical either way, but the `q_prefill` column therefore compares a cold-cache paged turn against a warm-cache reference turn and must not be read as a latency result. The cost table above is the one with every turn prefilled.

@acrosley 2026-08-19

## 2026-08-19 — The 0.6B pilot's testbed diagnosis confirmed: at 8B the governing failure is right there (8.71×), and `obey` is its worst query

Command: `scripts/stitch_train_8b.sh` step 1 — `python -m marathon.stitch_train eval --model Qwen/Qwen3-8B --items 48 --seed 9001 --gov-frac 0.5 --min-tokens 4000 --max-tokens 8000 --gen-tokens 32 --attn sdpa --base-only` (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, RTX 5090, ~21.3 GiB peak) · Model: `Qwen/Qwen3-8B` · Cost: $0.

The previous entry's pilot was negative and blamed its own testbed: at 0.6B on 3–5k sessions a governing edit cost only 1.50× a non-governing one, so the adapter had nothing to repair. That was a claim about *why* the run failed, and it needed testing before another epoch was spent. This is the gate — the base model on held-out seed 9001, 4–8k-token sessions, tuned columns skipped because an identity adapter only re-measures the base at double the price (`--base-only`).

**The failure class is exactly where it was said to be.**

```
bucket          metric               n      mean    median       p95       max  >.05
governing       base_stitch_kl      20    0.0373    0.0064    0.0845    0.4250     4
non-governing   base_stitch_kl      28    0.0043    0.0016    0.0174    0.0222     0
ALL             base_stitch_kl      48    0.0181    0.0030    0.0674    0.4250     4

governing/non-governing KL ratio:  mean 8.71x  (0.0373 / 0.0043)   median 4.03x  (0.0064 / 0.0016)
planted-fact ok:  governing 12/12 ref = base;  non-governing 16/18 ref = base
clean-context KL (identity adapter): 0.0000 on all 48 items, mean/median/p95/max alike
```

**8.71× on the mean** against the 144-session eval's ~9×, and 4.03× on the median against the ~3–4.5× that run showed — both reproduce at 48 items where 120 items at 0.6B gave 1.50× and 1.35×. Four governing items clear KL 0.05 and one clears 0.2 (0.4250); **no** non-governing item clears 0.05, with a maximum of 0.0222. So the diagnosis holds: the 0.6B result was about the model and the regime, not about stitched-KV consistency fine-tuning.

**The harness defect the previous entry admitted is fixed, and fixing it changed the picture.** `build_examples` took `item.queries[:k]`, and `kvshift_eval._queries` always puts `fact-at` first — which is why all 120 eval items in the 0.6B pilot asked the same question and `obey` never ran. It now rotates the entry point into the pool by session id, so a `k=1` population still covers every type. The first run with the fix says the query mattered:

```
qtype           n    base klmean   base klmedian
obey           11       0.0570          0.0080
fact-at        16       0.0103          0.0043
summarise       7       0.0057          0.0017
fact-after      8       0.0034          0.0009
fact-before     6       0.0014          0.0010
```

`obey` is the worst bucket by a factor of 5.5 on the mean, and the three worst items in the run are `governing/obey` (0.4250), `mid-governing/obey` (0.0845) and `governing/fact-at` (0.0674). That is worth flagging because it **cuts against** the 2026-08-18 2×2 entry, which found governing damage landing on the *fact* questions rather than on `obey` (governing × obey klmean 0.0161 vs governing × other 0.0304) and concluded the mechanism is not "the edited instruction steers the answer". On this smaller sample it is the instruction-following query that suffers most. Two readings are open and this run cannot separate them: 48 items with n=11 `obey` is small enough that one 0.425 outlier moves the mean a long way (the median ordering is much flatter — 0.0080 for `obey` against 0.0043 for `fact-at`), or the earlier run's `obey` cell was diluted because *every* item there also carried `fact-at`. Not resolved; do not quote the mean ordering as settled.

Also worth recording: **clean-context KL is 0.0000 across all 48 items**, exactly, which is the first confirmation on a real 8B model that the shared-`_clean_sequence` fix from the previous entry gives the damage metric a true zero rather than the ~0.005 bf16 floor it started with.

**Training did not complete, twice, and both failures were mine rather than the method's.** The first died immediately: `--attn` defaulted to `eager`, which materialises the full `[heads, q, kv]` fp32 score matrix — about 8 GB for a *single* layer at 8B and 8k tokens. `kvshift_eval` has always defaulted to `sdpa` for this reason and both paths honour the explicit additive mask the stitched forward passes; the default is now `sdpa`. The second died 81 items in with `RuntimeError: CUDA driver error: device not ready`, which is not a driver fault: `dmesg` shows `misc dxg: dxgk: dxgkio_make_resident: Ioctl failed: -12`, i.e. `-ENOMEM` — WSL's GPU paravirtualisation layer reports exhaustion as "device not ready" rather than as a clean torch OOM. Worth knowing for anything else run on this box.

The cause was a modelling choice with a memory price. `stitched_logits` kept the freshly computed span `E'` inside the autograd graph so `k_proj`/`v_proj` could learn what to *write* into the cache, not just how to *read* it — which costs a retained full-length K and V per layer, ~2.4 GB at 8B/8k, on top of the attention activations. It is now a flag (`--grad-prefill`) and off by default; the default runs the stitched prefill under `no_grad` and lets gradients flow only through the continuation, which is the arrangement Phase 3 was specified with (the reused KV is a constant). A test asserts the cheap path still produces a nonzero adapter gradient, because a trainer that silently no-ops is worse than a slow one.

The rerun of steps 2–4 was launched and **stood down within two minutes**: Track N started another `marathon.cold_eval --sessions 20 --turns 70` sweep at 04:12 and took 32 GB, and two tenants on one 32 GB card is how both runs get corrupted. The job was killed before it allocated anything.

**Verdict against the Phase 3 exit criteria: the gate is passed and the criteria remain untested.** Criterion 1 needs a governing/non-governing ratio ≤ 2× after tuning — the *before* number is now measured on the right regime at 8.71×, which is the number that run has to beat, but no adapter has been trained at 8B. Criteria 2 (clean-context KL ≤ 0.002) and 3 (no regression) likewise have their baselines and no treatment. What this entry settles is narrower and was the thing blocking everything else: **the experiment is now pointed at a population where the phenomenon exists**, the metric has a true zero, and the query pool is no longer degenerate.

Caveats: 48 items, one seed, one model. Cell sizes are 20 governing / 28 non-governing, so the four >0.05 items are four events and the one >0.2 is one; the mean ratio in particular rests on that single 0.4250 item, and dropping it would take the governing mean from 0.0373 to roughly 0.017 and the ratio from 8.71× to about 4×. The median ratio (4.03×) is the more robust of the two and is the one to hold the tuned run to. `--base-only` means the tuned and clean columns in that table are copies of the base by construction and carry no information. Session lengths were 4–8k as intended but were not recorded per item in this run.

@acrosley 2026-08-19

## 2026-08-19 — Stitched-KV fine-tuning works on the failure class it was built for (governing mean KL −43%, p95 −61%) and fails three of the five exit criteria

Command: `scripts/stitch_train_8b.sh` steps 2–4 — train 200 items seed 7001 lr 3e-5, held-out eval 60 items seed 9001, dependent-edit probe (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, RTX 5090, 24.4 GiB peak; training 200 items in 800 s) · Model: `Qwen/Qwen3-8B`, LoRA r=16 α=32 on q/k/v/o · Cost: $0.

Completes the previous entry, which passed the gate — the base model shows the governing failure at 8.71× on 48 items — but had no adapter. This is the adapter, on held-out seed 9001 at 60 items. **Base and tuned columns come from the same run over the same examples** (adapters off is the base model bit for bit), so the comparison carries no sampling difference.

```
bucket          metric               n      mean    median       p95       max  >.05
governing       base_stitch_kl      26    0.0372    0.0055    0.2344    0.4250     4
governing       tuned_stitch_kl     26    0.0211    0.0050    0.0905    0.2780     2
governing       tuned_clean_kl      26    0.0025    0.0018    0.0045    0.0152     0
non-governing   base_stitch_kl      34    0.0051    0.0024    0.0221    0.0390     0
non-governing   tuned_stitch_kl     34    0.0070    0.0035    0.0230    0.0394     0
non-governing   tuned_clean_kl      34    0.0023    0.0011    0.0062    0.0092     0
ALL             base_stitch_kl      60    0.0190    0.0034    0.0580    0.4250     4
ALL             tuned_stitch_kl     60    0.0131    0.0038    0.0355    0.2780     2

governing/non-governing ratio:  base  mean 7.25x  median 2.27x
                                tuned mean 3.01x  median 1.42x
planted-fact ok: 33/36 for reference, base and tuned alike (governing 15/15, non-governing 18/21)
```

**The method does something real, and it does it exactly where it was aimed.** On governing edits the mean falls 0.0372 → 0.0211 (**−43%**), p95 0.2344 → 0.0905 (**−61%**), max 0.4250 → 0.2780; items over KL 0.05 halve from 4 to 2 and items over 0.2 go from 2 to 1. 18 of 26 governing items improve. The two worst items in the run both improve substantially: `governing/obey` 0.4250 → 0.2780 and `governing/fact-at` 0.2344 → 0.0905. This is the first evidence in the project that the stale-attention failure is *trainable* rather than only avoidable by recomputation — and it is a tail effect, which is the part that matters, because the tail is what forces `reuse_plan` to refuse.

**And it is bought with a real cost.** Non-governing edits get *worse*: mean 0.0051 → 0.0070 (+37%), median 0.0024 → 0.0035 (+46%), with only 13 of 34 improving. The damage is a shift of the bulk rather than new failures — no non-governing item crosses 0.05 either way and the max barely moves (0.0390 → 0.0394) — but it is the regularisation set moving the wrong way despite being half the training mix. Clean-context drift is 0.0024 mean / 0.0015 median / 0.0152 max, with 23 of 60 items over 0.002.

**Against the Phase 3 exit criteria: two pass, three fail, and the failures are marginal rather than catastrophic.**

| criterion | target | measured | |
|---|---|---|---|
| 1. failure class closes | mean ratio ≤ 2×, no item > 0.2 | 7.25× → **3.01×**; one item still at 0.278 | **fail** (large move) |
| 2. clean context undamaged | clean KL ≤ 0.002 | **0.0024** mean (0.0015 median) | **marginal fail** |
| 3. no regression | non-governing within ~20% | **+37%** mean, +46% median | **fail** |
| 4. dep-instruction moves | toward the `first-512` level | 0.0133 → 0.0093, 0.0178 → 0.0104 | **pass** |
| 5. win kept | tokens forwarded unchanged | unchanged by construction | **pass** |

The honest verdict is **promising, not shippable**. Halving the tail on the one edit class that forces full recompute is the result this phase was looking for; doing it while making every other edit 37% worse is not a trade `reuse_plan` can take, because the entire point of the governing flag is that it refuses *only* the governing cases and leaves the other 191/269 alone.

**Dependent-edit probe** (single runs, not a distribution):

```
scenario          question           base_kl  tuned_kl  clean_kl  ref_ok base_ok tuned_ok
dep-instruction   lang-pipeline       0.0133    0.0093    0.0026    True    True     True
dep-instruction   lang-scheduler      0.0178    0.0104    0.0010   False   False    False
dep-anaphora      primary-key         0.0064    0.0095    0.0020    True    True     True
dep-anaphora      mission             0.0036    0.0020    0.0008    True    True     True
dep-anaphora      open                0.0187    0.0918    0.0025    None    None     None
dep-contradict    harbor              0.0057    0.0057    0.0007    True    True     True
dep-contradict    open                0.0137    0.0130    0.0051    None    None     None
```

Both `dep-instruction` questions improve by 30–40%, which is the scenario the adapter is for. But `dep-anaphora/open` degrades 0.0187 → 0.0918, a 5× regression on a scenario that was never broken — the same "hurts what already works" signature as the non-governing bucket, and on the open-ended question where free-running divergence is known to be generic. Note also that these `base_kl` values (0.006–0.019 klmean) are nowhere near the 0.3492 *first-token* KL the 2026-08-18 study recorded for `dep-instruction`: that was `kl_first` on a differently built session, and `klmean` over 32 tokens is a much gentler statistic. The two are not comparable, and the exit criterion should have named which one it meant.

**An ambiguity worth recording rather than smoothing over.** The training curve moves the wrong way: governing `stitch_kl` median 0.0065 (first half) → 0.0114 (second half), non-governing 0.0029 → 0.0033, `clean_kl` flat at 0.0013. Yet held-out governing KL improved 43%. The likely explanation is that the second half of the shuffled training set simply held harder governing items — per-item KL varies over two orders of magnitude here, so a 200-item split-half median is not a learning curve. But it could also mean the improvement is not attributable to training in the way assumed. The check is a held-out eval at a mid-training checkpoint, which this run did not save.

**Two infrastructure fixes made the run possible**, both diagnosed since the last entry. `--attn` defaulted to `eager`, which materialises the full `[heads, q, kv]` fp32 score matrix — ~8 GB for a single layer at 8B/8k — and died before the first item; it now defaults to `sdpa`, as `kvshift_eval` always did. Then training died 81 items in with `RuntimeError: CUDA driver error: device not ready`, which `dmesg` identifies as `misc dxg: dxgk: dxgkio_make_resident: Ioctl failed: -12`, i.e. `-ENOMEM` surfaced through WSL's GPU paravirtualisation layer rather than as a clean torch OOM. The cause was keeping the fresh span `E'` inside the autograd graph so `k_proj`/`v_proj` could learn what to *write* into the cache; that retains a full-length K and V per layer, ~2.4 GB at 8B/8k. It is now `--grad-prefill`, **off by default** — so the numbers above were produced with gradients flowing only through the continuation, which is the arrangement Phase 3 was specified with (the reused KV is a constant). Peak fell to 24.4 GiB and the run held ~4 s/item throughout.

That matters for how the result is read: **this is the weaker of the two available training signals.** Only `q_proj`/`o_proj` and the continuation tokens' `k`/`v` were trained; the model never learned to write a louder `E'`. The −43% mean and −61% p95 on the tail came from the cheap half of the method.

**What to try next, in order.** (1) Re-run with `--grad-prefill` at 4–6k sessions, where it fits — if the expressive half of the method is worth anything, that is where it shows. (2) Fix the regression before chasing the tail: the non-governing bucket getting worse while being half the training mix suggests the clean-context anchor is the wrong regulariser for it, and an explicit stitched-KV term on non-governing items (KL to teacher on cases that are already fine) is a more direct constraint. (3) Save mid-training checkpoints so the curve-versus-held-out ambiguity is resolvable. (4) Only then re-test the ratio.

Caveats: 60 held-out items, one seed, one epoch, 200 training items, one model. Cells are 26 governing / 34 non-governing, and the governing mean is dominated by two items over 0.2 — dropping the single 0.4250 item takes the base governing mean from 0.0372 to about 0.022 and the base mean ratio from 7.25× to ~4.3×, so the headline −43% rests substantially on how two items moved. The base ratios are themselves unstable with sample size: the same base model measured 8.71× mean / 4.03× median on 48 items and 7.25× / 2.27× on these 60, which means the ≤2× median target was nearly met by the *base* model and the median ratio is too noisy at this n to gate on. The `>0.2` criterion is 2 events before and 1 after. `planted-fact ok` grades the arg-max of the teacher-forced sequence, not a free-running greedy decode. No hyperparameter search was done — lr 3e-5 was chosen because 1e-4 misbehaved at 0.6B, not because it was tuned here.

@acrosley 2026-08-19

## 2026-08-19 — Why the connector broke on a paged session: a latched full-save and a staleness ratchet. Paged 14B now runs at exact-match 1.000 with byte-identical text

Track N measured the Phase 2 / Phase 1 composition failing: with a paged session (the oldest message demoted every turn, so a front-of-view edit on *every* turn) the shift connector scored exact-match 0.33 against 1.0 with it off, and p50 prefill 6.18 s with a 382 s outlier. My `save="full"` fix had been validated on two and three edits; this is what breaks at depth. Two independent bugs, one latency and one correctness.

**First, the CPU harness — which cleared the suspect I was most worried about.** `tests/test_paged_depth.py` models "the KV of the token at position p" as "the token id at position p", so re-rotation is exact by construction and any wrong fingerprint means wrong *coordinates*. It drives the connector's own decisions (`plan_load` / `plan_save`, factored out of the connector into `shift_store` for exactly this reason) through a block-granular prefix cache and a per-scheduler-step save loop, over 30 consecutive demote-style front edits. It passes: no corrupted position, no declined load, no refused save, no `covers()` miss. **Store coordinate bookkeeping is not the bug** — that rules out base/high-water drift, holes, and scheduler/worker mirror divergence, and it is worth having as a standing regression.

**Bug 1 — the full save latches, and every generated token re-gathers the whole prompt.** `_plan_save` runs on every scheduler step, decode steps included. `save="full"` forces `lo = 0`, so it was re-gathering the entire prompt across all 40 layers *once per generated token*. Phase 1 never saw it because the demo replies `ok` in two tokens; a paged eval answers real questions on 70 consecutive edit turns. The fix is to downgrade the flag to an ordinary incremental save the first time it is planned. That alone took p50 prefill on the paged shape from **6.18 s to 0.312 s**.

**Bug 2 — a staleness ratchet, and this one changes answers.** Reused KV attended to the text that preceded it when it was computed, and an edit turn *re-saves the span it just loaded*, so the reused vectors are never recomputed. In a paged session turn N's reuse carries N demotions of drift against a prefix that has been replaced by stubs. Measured on Qwen3-0.6B, 40 turns, a demotion every turn, teacher-forced so each turn is an independent comparison:

| `max_stale` | fact exact-match | control |
|---|---:|---:|
| unbounded | 0.200 | 1.000 |
| 4 | 0.500 | 1.000 |
| **1** | **1.000** | 1.000 |
| `repair_first=256`, unbounded | 0.200 | 1.000 |

`repair_first` does nothing here, consistent with the 2026-08-18 entry: repairing 256 tokens of an 1800-token span does not help when what changed is 97% of everything the span attended to. The fix is a ceiling on *consecutive* reused edit turns, after which one turn recomputes honestly. An append-only turn needs no reuse and resets the counter, so `max_stale=1` costs Phase 1's pattern nothing — isolated edits are still served entirely from reused KV.

**The failure mode is worth recording because it is not garbage.** The wrong answers were the *stub identifiers* (`00000004`, `0000000a`) rather than the planted code, with the control answering correctly from byte-identical text. Stale reused KV loses the attention competition to freshly computed tokens. Two measurement traps on the way: with the model's own replies fed back, the first divergence makes the two runs different *conversations* and every later mismatch is an echo — teacher forcing is required. And an open-ended prompt ("summarize what you have been told") diverges on paraphrase alone and reports damage where there is none; only exact-match on a planted fact, with the control at 1.000, is decision-grade.

**Qwen3-14B-FP8, ~8.3k paged window, 50 turns, demotion every turn from turn 12, against the same run with the connector off:**

| | mean | p50 | p90 | fact EM | text vs control |
|---|---:|---:|---:|---:|---|
| control (no reuse) | 0.954 s | 0.941 | 1.007 | 13/13 | — |
| **shift, fixed** | **0.574 s** | 0.871 | 1.027 | **13/13** | **0/50 turns differ** |
| shift, bug 2 unfixed | — | 0.312 | — | 5/13 | 31/50 differ |
| Track N, both bugs | — | 6.18 s | 12.1 | 0.33 | — |

Per-turn: the 19 reused turns average **0.171 s** against the control's 0.954 s (5.6×), and the 19 refresh turns average 0.977 s — the same as the control, which is what a refresh *is*. **Honest headline: 1.66× on the mean, not 5.6×.** The ceiling spends half the edit turns on an honest recompute, and that is the price of exact-match parity. All 50 turns' generated text is byte-identical to the no-reuse control, and Phase 1's own shape is unaffected — isolated edits at turns 12 and 20 on 14B still run at 0.187 s and 0.224 s against a control of 0.651 s and 1.287 s.

**Limits.** `max_stale=1` is the strongest setting short of disabling reuse, chosen because 4 still scored 0.500; between 1 and 4 is unmeasured and there may be a cheaper safe point on a less aggressive paging policy. The staleness counter counts *turns*, not drift — a better ceiling would measure how much of the reused span's attended context actually changed (the churn fraction), which would let benign repeated edits keep reusing and would probably beat 1.66×. This is a synthetic paged shape (a per-turn front demotion driven by `scripts/server_demo.py --demote`), not Track N's `cold.py` policy with its retriever and eviction, so the composition should be re-measured through `cold_eval` before the cold tier is called fixed. One model per shape, one run per cell, one window size.

## 2026-08-19 — Iteration 2: the hinge kills the regression and both arms clear every item over KL 0.05, but the ratio gate still misses on the median

Command: `scripts/stitch_arm.sh A` and `scripts/stitch_arm.sh B --grad-prefill` — train 200 items seed 7001 lr 3e-5 hinge 1.0, checkpoints every 50 with a 16-item held-out eval, then held-out eval n=120 seed 9001, then the dependent-edit probe (WSL2, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, RTX 5090; arm B peak 27.4 GiB, ~4 s/item; ~35 min per arm) · Model: `Qwen/Qwen3-8B` · Cost: $0.

Three changes since the last entry, all aimed at named failures in it. **(1) The do-no-harm hinge.** Iteration 1 cut governing KL 43% while *raising* non-governing 37%, despite non-governing being half the training mix — they were present but defenceless, starting at `klmean` ~0.005 against a governing item's ~0.037, so plain `stitch_kl` gave them almost no gradient. Non-governing items are now asked only not to get *worse* than the frozen base on that same item: `relu(klmean_student − klmean_base − slack)`, zero while the adapter is at least as good, costing one extra base stitched forward per item. **(2) `--grad-prefill`, memory-safe.** Iteration 1's numbers came from the cheap half of the method — only `q_proj`/`o_proj` and the continuation's `k`/`v` were trained, never what the fresh span `E'` *writes* into the cache. That is now available and bounded by a context cap rather than gradient checkpointing (checkpointing would re-enter the cache scatter during backward). **(3) Mid-training checkpoints**, to settle whether the improvement actually tracks training. Both arms run at **4–6k tokens** so they are comparable to each other; the base column is measured inside each eval on the identical items.

**Held-out, n=120 (57 governing / 63 non-governing), same items and same base for both arms:**

```
                          n     mean   median      p95      max   >.05   >.2
base    governing        57   0.0172   0.0070   0.0483   0.1480      3     0
  arm A governing        57   0.0109   0.0064   0.0353   0.0489      0     0
  arm B governing        57   0.0084   0.0068   0.0213   0.0281      0     0
base    non-governing    63   0.0046   0.0020   0.0128   0.0465      0     0
  arm A non-governing    63   0.0052   0.0025   0.0173   0.0460      0     0
  arm B non-governing    63   0.0054   0.0030   0.0158   0.0483      0     0
  arm A clean drift     120   0.0019   0.0015   0.0046   0.0077      -     -
  arm B clean drift     120   0.0024   0.0016   0.0059   0.0137      -     -

gov/non-gov ratio    base 3.74x mean / 3.51x median
                    arm A 2.10x mean / 2.53x median      arm B 1.56x mean / 2.26x median
planted-fact ok      72/72 for reference, base, arm A and arm B alike
governing improved   arm A 33/57   arm B 36/57
```

**The headline: every item over KL 0.05 is gone, in both arms.** The base has 3 governing items over 0.05 with a worst of 0.1480; arm A's worst is 0.0489 and arm B's is 0.0281. That is the number `reuse_plan` actually cares about — the tail is what forces it to refuse governing edits, and on this population the tail is now inside the band where non-governing edits already live.

**The hinge worked.** Non-governing regression falls from iteration 1's **+37% mean / +46% median** to **+13% / +25%** (arm A) and **+17% / +50%** (arm B), and arm A now passes the ≤20% mean bar outright. Clean drift also improved, arm A to **0.0019**, which clears the 0.002 gate for the first time. Planted-fact retrieval is 72/72 everywhere — reference, base and both adapters — with no item lost.

**Arm B (the expressive half) is clearly the stronger learner, and the checkpoint curves show it is not close:**

```
checkpoint (16 held-out items)      50       100       150       200
arm A governing   base 0.0446   0.0507    0.0456    0.0366    0.0226
arm B governing   base 0.0446   0.0238    0.0193    0.0242    0.0123
```

Arm A is still *worse than the base* at checkpoint 50 and does not clearly beat it until 150; arm B is at half the base's KL by checkpoint 50 and ends at 0.0123. Letting gradients reach what `E'` writes into the cache is worth roughly a 2× head start throughout, and 200 items of the cheap signal buys about what 50 items of the expressive one does. This also answers iteration 1's unresolved ambiguity — where split-half training medians moved the wrong way while held-out KL improved, with no way to tell which was real. With actual intermediate measurements the held-out curve is **non-monotone but genuinely downward** (arm B: 0.0238 → 0.0193 → 0.0242 → 0.0123), so the improvement does track training; the split-half training median remains the misleading statistic, not the result.

**Against the pre-registered gates** (all `klmean`, both statistics reported, n=120 as required):

| criterion | target | arm A | arm B |
|---|---|---|---|
| 1. failure class closes | ratio ≤ 2× mean **and** median; no item > 0.2 | 2.10× / 2.53× — **fail** | **1.56×** / 2.26× — **fail (median)** |
| 2. clean context | ≤ 0.002 mean | **0.0019 — pass** | 0.0024 — fail |
| 3. no regression | non-gov within 20%, mean and median | +13% / +25% — **fail (median)** | +17% / +50% — fail |
| 4. dep-instruction | ≥ 30% fall vs same-run base | 0.0133→0.0315, 0.0178→0.0229 — **fail** | 0.0133→0.0271, 0.0178→0.0120 — **fail** |
| 5. win kept | tokens forwarded unchanged | pass | pass |

**Neither arm passes, and the two fail differently: arm B buys the tail, arm A protects the collateral.** Arm B has the better ratio (1.56× mean, and it is the only run in the project to clear the ≤2× mean bar) and the better tail (max 0.0281); arm A has the better clean drift (0.0019, passing) and the smaller non-governing median regression. Neither dominates, which is itself informative — the expressive half is a stronger lever *and* a blunter one, and the honest next move is arm B's signal with arm A's restraint (a stronger hinge weight, or early stopping around checkpoint 100–150 where arm B's non-governing had drifted less).

**The `dep-instruction` probe got worse in both arms and this is the clearest negative result here.** `lang-pipeline` goes 0.0133 → 0.0315 (arm A) and → 0.0271 (arm B); only arm B's `lang-scheduler` improves (0.0178 → 0.0120). `dep-anaphora/open` also degrades badly in both (0.0187 → 0.0732 / 0.0133). The likely reason is distribution: the probe scenarios are 20-turn sessions of ~600-token filler messages, materially longer and structurally different from the 4–6k training sessions, so the adapter is being asked to generalise off-distribution — exactly the overfitting-to-synthetic-edits risk that was written down before any of this ran. It is not evidence the method fails on governing edits; it is evidence the adapter learned something narrower than "read stitched caches correctly".

**An important caveat about the regime, which cuts against reading these ratios as progress over iteration 1.** Both arms ran at 4–6k, where the *base* failure is much milder: base ratio 3.74× mean / 3.51× median here, against 7.25× / 2.27× at 4–8k (n=60) and 8.71× / 4.03× at 4–8k (n=48). The gate is easier to approach in this regime because there is less to fix. Arm B's 1.56× is a real number against a real same-run base, but it is not directly comparable to iteration 1's 3.01×, and the ≤2× target was chosen against a ~9× base. Any claim that the gate is nearly met has to be re-made at 4–8k.

**What to try next.** (1) Arm B's signal at a higher hinge weight (2–4) or with early stopping near checkpoint 100–150, to keep the tail win without the non-governing median drift. (2) Re-run the winner at **4–8k**, where the base failure is 2× larger, before believing the ratio. (3) Add the probe scenarios' *shape* to the training mix — longer sessions, filler-message structure — since the current failure there is plainly a distribution gap and the criteria treat it as a gate. (4) Only then consider whether `reuse_plan` could downgrade governing edits from `repair` to `reuse`; on the tail evidence alone (0 items over 0.05 in either arm) that conversation is now worth having, but not on one seed.

Caveats: one seed pair, one epoch, 200 training items, one model, n=120 held out. The mid-training curve uses a 16-item slice, so its individual points are noisy — the 0.0242 at arm B checkpoint 150 is 16 items. Cells are 57/63, and the base tail is 3 events. `planted-fact ok` grades the arg-max of the teacher-forced sequence, not a free-running greedy decode. The hinge costs one extra base forward per non-governing item, so arm timings are not comparable to iteration 1's. And the dep-probe rows remain single runs on hand-built scenarios, not a distribution.

@acrosley 2026-08-19

## 2026-08-19 — Re-measured through `cold.py`: reuse composes (2.6×), the *refresh* turn costs 48 s, and the churn ceiling buys the same accuracy for half the price

Command: `python -m marathon.cold_eval --generate-history --turns 70 --active-window 8192 --threshold 0.2` with `--conditions` / `--max-stale` / `--max-churn` per row (`scripts/cold_eval.sh`) · Model: `Qwen/Qwen3-14B-FP8`, vLLM 0.27.1, retriever `all-MiniLM-L6-v2` on CPU · history p50 25.0k / max 35.0k tokens.

The previous entry asked for exactly this: `max_stale` was validated on a synthetic per-turn front demotion (`server_demo.py --demote`), not on `cold.py` with its retriever, stubs and eviction. Re-measured on the real policy, with every history turn actually prefilled.

**N, honestly.** Sessions are seeded and the corpus frozen, so a session id is byte-identically the same session in every condition. Collected: cold-recall N=10, cold-nostale N=4, cold-churn0.5 N=4, cold-stale1 N=3 (the GPU was needed by another track). **The table is the matched subset — sessions 0, 1, 2 — of all four**, 213 history turns each. Full-N runs agree to within small-sample noise (cold-recall em_all 0.817 at N=10 vs 0.889 here).

| condition | connector | prefill p50 | prefill mean | reuse turn | refresh turn | refresh frac | active max | em_old | em_recent | em_all | promo recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold-recall | off | 0.648 | **0.537** | — | — | 0 | 8060 | 0.889 | 0.889 | **0.889** | 0.941 |
| cold-nostale | on, unbounded | 0.192 | **0.173** | 0.207 | — | 0 | 8060 | 0.111 | 0.111 | **0.111** | 0.941 |
| cold-stale1 | on, `max_stale=1` | 0.479 | **19.170** | 0.523 | 48.29 | 0.393 | 8060 | 0.333 | 0.556 | **0.444** | 0.941 |
| cold-churn0.5 | on, `max_churn=0.5` | 0.290 | **8.520** | 0.405 | 36.92 | 0.224 | 8060 | 0.222 | 0.667 | **0.444** | 0.941 |

**Reuse composes; the answer to "does Phase 2 meet Phase 1" is yes for the mechanism and no for the system.** Turns actually served from re-rotated KV cost **0.207 s** against the connector-off baseline's 0.537 s mean — a genuine **2.6×** — and the whole unbounded condition runs at 0.173 s mean, **3.1×** faster than prefix caching on the same paged workload. The k+1 phase driver is not a problem either: on the unbounded run, 6-phase turns average 0.242 s and 1-phase turns 0.061 s. Paging is unaffected by any of this — active-window max is 8060 in every row and promotion recall 0.941 in every row, because retrieval is deterministic and independent of the KV path.

**What breaks it is the refresh turn, and it is a serving bug, not a policy cost.** A refresh sets `loads=[]`, so `phases` is empty and the turn takes the ordinary single-request `generate(..., save=self.reuse)` branch — the same prefill cold-recall does in 0.537 s, plus a connector save. It costs **48.3 s**. Clean isolation, identical config (1 session, 40 turns, cold-shift):

```
max_stale=999 (never refreshes)     16.7 s
max_stale=1   (the default)        731.6 s      44x
```

Early append-only turns save in 0.081 s, so saving as such is cheap; it is specifically a save *after* a reused turn, and it runs at 114 W of 575 W — memory bound, i.e. the store is being rewritten rather than extended. The previous turn re-saved `save="full"` in its coordinates and the refresh then writes into coordinates that have shifted again. Same family as the latched full-save, on the refresh path rather than the decode loop.

**The churn ceiling works, and is strictly better than counting turns.** `max_churn=0.5` scores **the same exact-match as `max_stale=1` (0.444) at 2.25× less mean prefill** (8.52 s vs 19.17 s), because it refreshes on 0.224 of turns instead of 0.393: it lets a run of edits that barely disturbs the reused span's prefix keep reusing, where the turn counter refreshes on a schedule regardless. `shift_store.churn_tokens` returns the tokens in front of the deepest reused segment that the plan does *not* reuse; the server accumulates that across consecutive reused turns and refreshes at `accumulated / span_len > max_churn`. CPU calibration over six sessions predicted the refresh fractions within noise (0.253 predicted vs 0.224 measured at 0.5; 0.500 vs 0.393 for `max_stale=1`). **On present evidence the churn ceiling should be preferred to the turn counter** — same accuracy, half the bill — though that is a comparison between two losing configurations, see below.

**Quality tracks the refresh rate, and none of it reaches the baseline.** 0.111 unbounded → 0.444 at both bounded settings → 0.889 connector-off. Staleness damages this workload exactly as it did the synthetic one, but `max_stale=1` does *not* restore parity here the way it did there (1.000 against a control of 1.000). The difference is the shape: the synthetic demotes once per turn against a mostly-static prefix, while `cold.py` demotes 3.19 entries per turn and promotes up to 2 more, so one honest recompute every other reused turn cannot keep up with the drift. **There is currently no setting of either ceiling that is both as accurate as connector-off and faster than it**, and the honest summary is that the cold tier still serves best with the connector off.

**A reuse-plan misclassification found and fixed on the way.** `moved` keyed on "this entry's index changed", which is true of every entry after a deleted one — so a cold-tier *eviction* (dropping a stub from the view is a pure, order-preserving line deletion) was classified as 14 relocated entries and reused nothing at all. The test is ordering, not index equality: read in destination order, the matched source indices are strictly increasing exactly when nothing was reordered. Fixed in `_segments` and in `plan()`'s `relocated` (which was additionally forcing `policy="repair"` on governing entries that had merely been pushed along); genuine reorders fall back to the old, stricter rule. Measured over four 70-turn paged sessions — at an 8192 window eviction never fires and the fix is worth nothing, but at 4096 the old rule discarded **43%** of the turns that can reuse, and at 2048, **84%**. It does not affect the table above; it is what makes tighter windows viable.

**One interaction fixed.** `turn(..., generate=False)` advanced the staleness counter although such a turn saves and loads nothing, so a replayed history forced spurious refreshes on the real turns after it. Both the counter and the churn accumulator are now left untouched on a non-generating turn.

**Limits.** N=3 matched (N=4 and N=10 per condition available and consistent). 17 of the 18 fact questions in the subset had their answer cold, so em is over 18 questions per condition and the 0.444-vs-0.444 equality is "not distinguishable at this N" rather than a measured tie — the *cost* difference between the two ceilings is the solid result. One model, one window (8192), one threshold pair (0.5, plus 0.2 from CPU calibration only — dropped from the GPU run because it refreshes *more* than `max_stale=1` and so is the expensive end of the curve). `max_churn` between 0.5 and unbounded is unmeasured, and that is where a cheaper safe point would be once the refresh bug stops dominating the bill.

## 2026-08-19 — The paged stall is store *allocation*, not the save path — and on the real cold tier the composition is still wrong at 0.5 exact-match

Track N measured refresh turns at 41.5 s mean against reuse turns at 0.90 s and plain appends at 0.08 s, and hypothesised a per-step full re-gather in the post-reuse incremental save. That hypothesis is wrong, and the CPU harness says so cheaply.

**The save path is innocent.** Instrumenting `tests/test_paged_depth.py` to count store writes per turn, over a 20-turn paged session:

| turn kind | positions written per turn |
|---|---:|
| plain append | ~265 (O(new tokens)) |
| reuse (`save="full"`) | ~1035 (one pass over the prompt) |
| refresh (`save=True`) | ~1035 (one pass over the prompt) |

A refresh turn writes the prompt exactly once — the same volume as a reuse turn, no 40× amplification and no per-decode-step latch. Two regression tests now pin this (`test_refresh_turn_does_not_rewrite_the_store_many_times_over`, `test_no_turn_writes_the_store_more_than_once_over`, asserting every turn writes under 1.5× its own prompt). Nothing in the save path was changed, because nothing in it was broken.

**The bug is that the store reallocated its buffers on every single turn.** `SLAB = 16384` was the *fixed first allocation*, not a floor, so a session holding 235 tokens still claimed a 16384-token slab. With Track N's `store_tokens=24576`, two sessions (32768) cannot coexist, so each session's `_grow` evicted the other and both reallocated all 40 layer buffers every turn. Measured on the CPU harness, three sessions × 14 turns:

| | grows | evictions |
|---|---:|---:|
| before | **42** — one per turn, every one from `capacity=0` | 41 |
| after | **3** — one per session, for life | 0 |

On 14B a slab is 2.7 GB, so that is 2.7 GB allocated, zeroed and freed *per turn*, which is what "41 s at 114 W, memory-bound" actually looks like. The fix is one line in `_grow` — `size = max(self.slab, entry.capacity * 2)` instead of `size = self.slab`, with `SLAB` becoming a 2048-token floor. Growth stays geometric, so the 2026-08-19 slab entry still holds: that entry blamed *fixed 1024-token steps*, and doubling is what fixed it — the large first allocation was never the part that mattered. Reverting the two lines makes the new test fail with "35 evictions: the sessions are thrashing the budget". Track N's own log corroborates the diagnosis: `used_tokens: 16384, sessions: {'cold-shift-3': 7744}` — one session occupying the entire budget while holding 7.7k tokens.

**Ruled out from that log:** zero preemptions and a healthy 12.66 GiB / 82,928-token KV cache, so it is not cache starvation; zero refused saves, zero "no stored KV", zero in-flight-writer conflicts. All 214 "declining reuse" lines are the benign `nothing left after block alignment` on two-phase warm-up requests.

**GPU verification, Qwen3-0.6B, 40 paged turns — criterion met.** Refresh-turn mean **0.034 s** against a connector-off control's plain-turn mean of 0.037 s; reuse turns 0.030 s; fact exact-match 10/10 both sides.

**GPU verification, Qwen3-14B-FP8 on Track N's exact config** (real `cold.py` paging at `--active-window 8192`, `store_tokens 24576`, `gpu_util 0.93`, 40 turns, `max_stale=1`), against a connector-off control on the same config:

| turn kind | n | mean | max |
|---|---:|---:|---:|
| plain (in reuse run) | 13 | 0.139 s | 0.174 |
| reuse | 14 | 0.311 s | 0.933 |
| refresh | 13 | **6.159 s** | 26.856 (min 1.000) |
| control — every turn | 40 | 0.502 s | 1.014 |

Refresh turns are no longer uniformly pathological — but the mean hides a shape. Turns 14–24 cost 7–27 s; from turn 26 on they settle at 1.0–1.4 s, i.e. an ordinary full prefill of an 8k window. The slow ones are the turns that happen to trigger a growth step. **Growing at all is what hurts**: the store is carved out of a GPU vLLM has already filled to `gpu_memory_utilization=0.93`, so a realloc must hold the old and new buffers at once with no headroom and the allocator falls back on synchronising and returning blocks to the driver. The remaining fix is to size the store to the workload up front, or to leave the card headroom — not to change the growth policy again. I tried "a sole session takes the whole budget", and it is wrong: the first session is always sole at allocation time, so it fills the budget and we are back to evicting everyone.

**And the correctness result is negative, which matters more than the latency one.** On the real cold tier the connector scores **fact exact-match 5/10 against the control's 10/10**, with 5 of 40 turns differing in text — despite `max_stale=1`, which held at 10/10 on the synthetic shape. The difference is the workload: my synthetic demotion produces a *single* reused segment per turn, while the real policy also promotes, so a turn carries 5–6 segments across 6 phases with relocations among them. The staleness ceiling was tuned on the easy shape. **Phase 2 × Phase 1 is still not correct on the real paging policy**, and the next thing to measure is which of promotion / eviction / multi-segment phasing is responsible — the per-turn records now carry `promotions`, `demotions`, `segments` and `deltas`, so that is a read of the existing JSON rather than a new run.

**Limits.** One run per cell; the 14B reuse and control runs are separate engine invocations (the first attempt's control died at engine-core init under `util 0.93` back-to-back, which is why `paged_depth.sh` now sleeps and deletes the stale JSON before the second run — a stale file had silently produced a 0.6B-vs-14B comparison). The 0.6B result is on the synthetic shape, not `cold.py`. The eviction-thrash fix is verified on CPU and by inference on GPU; no run isolates a growth event directly.

@acrosley 2026-08-19

## 2026-08-19 — The paged 5/10 is stale attention, not a coordinate bug: all damage is on reuse turns, and it tracks how much of the view is stubs

Three CPU results on the 14B paged run's own records, plus a fingerprint harness extended to `cold.py`'s real policy.

**1. Every wrong turn is a reuse turn.** Splitting the 40-turn run by turn kind, against the connector-off control on identical (teacher-forced) histories:

| turn kind | n | turns diverging from control | scored turns |
|---|---:|---:|---:|
| plain | 13 | **0** | 3 |
| reuse | 14 | **5** | 7 |
| refresh | 13 | **0** | 0 |

Refresh turns are byte-identical to the control, every one — the staleness ceiling's recompute does exactly what it is supposed to. All damage is in re-rotated KV.

**2. The headline 5/10 was measuring the wrong thing.** The probe asks its question every 4th turn, and `max_stale=1` makes reuse and refresh alternate, so from turn 13 on *every scored turn lands on a reuse turn* — 7 scored reuse turns, 3 scored plain, **0 scored refresh**. The ceiling can never protect a scored turn, so 5/10 is "how often is a reuse turn right", not "how good is the system". The other 30 turns carry no signal at all: their replies are canned (`Understood.` / `ok`), which is also why "5 of 40 diverged" and "5 of 10 scored" are the same five turns. Any future paged eval has to sample scored turns independently of the reuse/refresh phase.

**3. What predicts a wrong reuse turn: how much of the view has become stubs.**

| turn | hit | segments | phases | promos | cold msgs | reused % | max abs delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 13 | **MISS** | 1 | 2 | 0 | 1 | **91.7%** | 566 |
| 17 | hit | 6 | 4 | 2 | 13 | 74.5% | 5528 |
| 21 | hit | 6 | 5 | 2 | 21 | 75.1% | 5535 |
| 25 | **MISS** | 6 | 6 | 2 | 31 | 73.1% | 4374 |
| 29 | **MISS** | 6 | 6 | 2 | 39 | 74.6% | 4920 |
| 33 | **MISS** | 6 | 6 | 2 | 49 | 72.8% | 3749 |
| 37 | **MISS** | 6 | 6 | 2 | 57 | 74.4% | 4302 |

Promotions do not predict it (2 on every turn, hit and miss alike). Segment count does not (6 on both). Reused fraction does not (74.5% hits, 74.6% misses). **Cold count does**, monotonically and with a clean threshold between 21 and 31 — and turn 13 is a second regime, the one turn that reuses 92% of the prompt off a single front demotion. Both are the same underlying quantity: how much of the text a reused span attended to has since been replaced. That is churn measured in tokens, not turns, and the current ceiling counts turns.

**4. The coordinates are exact — the harness says so, and it can prove it is looking.** `tests/test_paged_depth.py` now drives `marathon.cold`'s real policy (demotions, retriever promotions, stub evictions, multi-segment phasing) through the token-id-as-KV fingerprint, teacher-forced. Over 30 turns at an 1500-token window: 54 demotions, 44 evictions, 43 promotions, plans up to 6 segments, and **zero corrupted positions, zero declined loads, zero refused saves**; a 40-turn run at a 900-token window forces evictions specifically and is also clean. Injecting a one-block error into `plan_load`'s delta produces **1181 corrupted positions**, so the clean result is a measurement rather than a blind spot. Every position holds its own token under the full policy: the connector is putting KV exactly where it belongs, and the answer damage is stale attention.

**5. Pre-sizing the store, so growth never happens on a full GPU.** `ShiftStore` takes a `session_cap`: when the caller knows a session's ceiling — a server with a bounded active window does — the first save allocates that much and no later save grows. It is a floor, not a ceiling (a session that outgrows it still grows geometrically), and it is off by default so multi-session budgets keep sharing. `MarathonServer` passes `active_window + max_tokens + 256` through `kv_connector_extra_config["session_tokens"]`. Three CPU tests cover it: one allocation for a session filling its window turn by turn, growth still available past the cap, and geometric behaviour unchanged when it is off. This is aimed at the 7–27 s growth-step turns measured at `gpu_util=0.93`; it is unverified on GPU.

**Limits.** The predictor table is seven reuse turns from one run — cold count and turn index are confounded (both rise monotonically), so "stub fraction" is a hypothesis consistent with the data, not an isolated cause. The fingerprint model treats re-rotation as exact by construction, so it can only ever find *coordinate* errors; it cannot see attention damage, which is precisely the thing now suspected. Segment spans are now recorded per turn so a real churn metric can be computed from the next run's JSON without another GPU run. Pre-sizing and the churn-based ceiling are both untested on hardware; GPU verification is queued.

## 2026-08-19 — Iteration 3 at 4–8k: the gates fail on both hinge weights, and iteration 2's promising ratios turn out to be mostly a regime artefact

Command: `scripts/stitch_train_8b.sh` (base-only n=120, then w=2) and the same with `--skip-basecheck --preserve-weight 4` (w=4) — train 200 items seed 7001 lr 3e-5, `--grad-prefill` capped at 6000 tokens, `--standing-frac 0.34`, checkpoints every 50 with a 24-item held-out eval, checkpoint chosen by the pre-registered rule, then held-out eval n=120 seed 9001 and the dependent-edit probe (WSL2, `~/marathon-venv`, torch 2.13.0+cu130, transformers 5.15.0, sdpa, bf16, RTX 5090; peak 26.86 GiB both arms) · Model: `Qwen/Qwen3-8B`, LoRA r=16 α=32 on q/k/v/o · Cost: $0.

Everything here was pre-registered in [phase3-design.md](phase3-design.md) before the GPU was touched: the 4–8k regime, the hinge-weight ordering (`w=2` first, `w=4` if time), the checkpoint-selection rule, and the new `standing-governing` bucket. That matters, because the result is negative and the temptation to re-read it favourably is exactly what pre-registration removes.

**The base at 4–8k, n=120 held out — the number the gate is actually against:**

```
bucket          metric            n      mean    median       p95       max  >.05  >.2
governing       base            46    0.0284    0.0050    0.0328    0.4910     2     2
standing-gov    base            14    0.0061    0.0030    0.0155    0.0172     0     0
non-governing   base            60    0.0054    0.0024    0.0157    0.1031     1     0

base gov/non-gov ratio   5.26x mean / 2.11x median     (+std 4.29x / 1.86x)
```

**Both arms, same 120 items, same in-run base, checkpoint chosen by the rule:**

```
                          n     mean   median      p95      max   >.05   >.2
base    governing        46   0.0284   0.0050   0.0328   0.4910     2     2
  w=2 (step150)          46   0.0234   0.0050   0.0339   0.4532     2     2
  w=4 (step200)          46   0.0233   0.0059   0.0504   0.4496     3     2
base    standing-gov     14   0.0061   0.0030   0.0155   0.0172     0     0
  w=2                    14   0.0053   0.0042   0.0110   0.0111     0     0
  w=4                    14   0.0048   0.0027   0.0113   0.0135     0     0
base    non-governing    60   0.0054   0.0024   0.0157   0.1031     1     0
  w=2                    60   0.0059   0.0022   0.0183   0.0956     1     0
  w=4                    60   0.0051   0.0029   0.0142   0.0412     0     0
  w=2 clean drift       120   0.0020   0.0014   0.0055   0.0130     -     -
  w=4 clean drift       120   0.0022   0.0015   0.0068   0.0122     -     -

gov/non-gov ratio    base 5.26x mean / 2.11x median
                      w=2 3.98x / 2.26x        w=4 4.59x / 2.02x
governing improved    w=2 24/46                w=4 22/46
planted-fact ok       64/65 for reference, base, w=2 and w=4 alike
grad-prefill          78/200 items expressive (cap 6000), 0 OOM fallbacks, peak 26.86 GiB
```

**Against the pre-registered gates:**

| criterion | target | w=2 | w=4 |
|---|---|---|---|
| 1. failure class closes | ratio ≤ 2× mean **and** median; no item > 0.2 | 3.98× / 2.26×, 2 items > 0.2 — **fail** | 4.59× / 2.02×, 2 items > 0.2 — **fail** |
| 2. clean context | ≤ 0.002 mean | **0.001999 — pass by 1e-6** | 0.0022 — **fail** |
| 3. no regression | non-gov within 20%, mean and median | **+8.9% / −6.6% — pass** | −5.8% / **+21.6%** — **fail (median)** |
| 4. dep-instruction | ≥ 30% fall vs same-run base | 0.0133→0.0198, 0.0178→0.0170 — **fail** | 0.0133→0.0310, 0.0178→0.0161 — **fail** |
| 5. win kept | tokens forwarded unchanged | pass | pass |

**Neither arm passes, and the headline is that iteration 2 was measuring an easier problem than it looked.** At 4–6k the base ratio was 3.74× and arm B reached 1.56× mean with *every* item over KL 0.05 eliminated. At 4–8k the same method reaches 3.98× against a 5.26× base, the two items over KL 0.2 are still there afterwards, and w=4 actually *adds* a third item over 0.05 and raises p95 (0.0328 → 0.0504). The tail did not close. The pre-registration called this risk in advance — "any claim that the gate is nearly met has to be re-made at 4–8k" — and the answer is that it is not nearly met.

**The hinge weight is not the missing knob.** Doubling it from 2 to 4 moved the non-governing *mean* the right way (+8.9% → −5.8%, i.e. w=4 actually improves the cases it is only asked to protect) and the *median* the wrong way (−6.6% → +21.6%), while clean drift went 0.0020 → 0.0022 and the governing tail got worse. "Arm B's signal with arm A's restraint" was the hypothesis; more restraint bought neither.

**The checkpoint rule worked, and it cost us the better tail on purpose.** For w=2 the rule selected step 150 and *rejected* step 200, whose governing p95 was better (0.2077 vs 0.2627) but whose clean drift on the mid-slice was 0.0022, over the 0.002 budget. That is the rule doing its job — criterion 2 is a gate, not a preference — but the reported w=2 tail is not the best tail the run produced. For w=4 the rule selected step 200. Both curves are non-monotone and genuinely downward on the governing mean (w=2: 0.0556 → 0.0600 → 0.0388 → 0.0307), consistent with iteration 2.

**Two findings that are more useful than the gate verdict.**

**(1) The base measurement is not reproducible across runs, and the mean ratio rests on the irreproducible part.** The base-only eval at step 1 and the base column of the w=2 eval are the same model on the same 120 items from the same seed. 87 of 120 rows are bit-identical; 33 differ, and the largest disagreements are enormous — item 57 `fact-at` reads 0.1351 in one run and **0.4910** in the other; item 6 reads 0.0018 and 0.1031. Mean |diff| is 0.0062, against a governing mean of ~0.026. The mechanism is bf16 non-determinism flipping the teacher's greedy token, after which the whole 32-token teacher-forced sequence differs. Since the governing mean is carried by two or three tail items, **cross-run mean-ratio comparisons at this sample size are inside the noise** — which retroactively weakens every cross-iteration ratio claim in this phase, including iteration 2's 3.74× → 1.56×. The *in-run* base/tuned comparison is unaffected and remains sound: `evaluate` computes the teacher once per item with adapters off and pairs both columns against it. Any future gate should quote the paired per-item delta, not two independently measured means.

**(2) The `dep-instruction` distribution gap was not the explanation.** `standing-governing` put probe-shaped sessions — the probe's own filler generator, the instruction in the system prompt or an early governing user turn, open-ended questions with no forced prefix — into training and the held-out eval at 4–8k, at 34% of the governing half. The bucket behaves well (base mean 0.0061, nothing over 0.05, and both arms improve it slightly) and criterion 4 **still fails in both arms**, with `lang-pipeline` getting worse in each. So the iteration-2 hypothesis that the probe regressed because it was off-distribution is not supported: the bucket that closes the distribution gap barely exhibits the failure in the first place (0.0061 against core governing's 0.0284), so it carries little gradient and cannot teach the probe's behaviour. The probe's `dep-instruction` scenario is measuring something the synthetic governing population does not contain, and that is now a question about the population rather than about the training regime.

**Infrastructure, since it cost an hour of the window.** The first launch produced a 0-byte log: `nohup … &` inside `wsl.exe -- bash -c` does not survive the parent exiting, and PowerShell had separately eaten the redirect. The second attempt died in training with `RuntimeError: CUDA driver error: device not ready` — WSL's ENOMEM again — at a `--grad-prefill` cap of 8600 where every item was expressive, having printed `longest item 8560 tokens, estimated full-length K/V 3.53 GiB (3 copies at 144 KiB/token)` and peaked at 28.8 GiB by checkpoint 50. The pre-registered memory estimate was accurate; the margin it predicted (~28–29 GiB on a 32 GiB card) was simply too thin. Two fixes: the cap dropped to 6000 (iteration 2's proven 27.4 GiB envelope), and the OOM fallback now recognises WSL's mislabelled form. That second one is a correction to this project's own pre-registration, which explicitly declined to catch plain `RuntimeError` on the grounds that a poisoned context must not be retried — right in principle, wrong on this box, and it cost the run. It now matches the known ENOMEM signatures, retries once, and re-raises if the retry also fails; a non-exhaustion `RuntimeError` still propagates untouched.

**The cost of that fix is a confound worth stating plainly.** At a 6000-token cap only **78 of 200** training items kept the fresh span in the graph, against 200/200 in iteration 2's arm B. So iteration 3 is not "arm B at 4–8k" — it is a 39%-strength version of arm B at 4–8k. The comparison to iteration 2 is confounded by regime *and* by expressive coverage, and neither confound can be separated from this data.

**Where this leaves the phase.** Three iterations in, the method reliably cuts the governing mean (43%, then 51%, now 18%) and has never met criterion 1. The most valuable next step is not another hyperparameter: it is fixing the measurement, because finding (1) says the current gate cannot resolve the differences being argued over. Concretely — quote paired per-item deltas with a bootstrap CI rather than a ratio of means; raise n or seed-average the base; and decide whether `klmean` on a teacher-forced sequence whose *first token* can flip is the right target at all. After that, the open modelling question is whether 8k-token expressive training (which needs either more VRAM or the chunked `E'` forward that was deferred here) closes the tail that 39% coverage did not.

Caveats: one seed pair, one epoch, 200 training items, one model, n=120 held out, cells 46 governing / 14 standing / 60 non-governing. The base tail is 2 events and the governing mean is dominated by them — see finding (1) for why that is worse than it usually is. The mid-training curve uses a 24-item slice, so the selection rule is applied to noisy p95 estimates. `w=4`'s base column is the same run's base, but its *selected checkpoint* differs from `w=2`'s (step 200 vs 150), so the two arms differ in training length as well as in hinge weight and are not a clean one-variable comparison. `planted-fact ok` grades the arg-max of the teacher-forced sequence, not a free-running greedy decode. The dep-probe rows remain single runs on hand-built scenarios, not a distribution.

@acrosley 2026-08-19

## 2026-08-20 — Phase 2 × Phase 1 does not compose: on a paged workload, reuse turns are fast and wrong, refresh turns are right and slow

The churn ceiling landed (`max_churn`, main), the two measurement bugs from yesterday are fixed — scored turns are now asked on *two consecutive* turns so they cannot phase-lock onto the reuse/refresh alternation, and the store is pre-sized from the active window — and the composition question can finally be answered. It is a no.

**First, a correction to yesterday's entry.** I reported that cumulative cold count separated hit from miss with a boundary between 21 and 31. Re-checking every metric reconstructable from that run against the labels:

| metric | hits | misses | separates? |
|---|---|---|---|
| sum abs delta | 7213, 7213 | 566, 7747, 7747, 7132, 7132 | no |
| max abs delta | 5528, 5535 | 566, 4374, 4920, 3749, 4302 | only backwards |
| churn / prompt | 0.944, 0.916 | 0.071, 1.026, 0.996, 0.956, 0.928 | no |
| reused fraction | 0.745, 0.751 | 0.917, 0.731, 0.746, 0.728, 0.744 | no |
| cold count | 13, 21 | **1**, 31, 39, 49, 57 | **no** |

Cold count does not separate: turn 13 misses at cold=1. Yesterday I excluded it as "a second regime", which was fitting the outlier away. `max abs delta` separates only in the physically implausible direction (hits displaced *further*), which on two-versus-five points is noise. **No metric separates, and with 2 hits in 7 the honest reading is that reuse was unreliable throughout.** No threshold was fitted for this run; 0.25 and 1.0 were chosen to bracket the ~0.37 churn fraction a single paging turn carries, so the sweep measures the frontier rather than validating a fitted number.

**Qwen3-14B-FP8, active window 8192, 40 turns, paged turns (13–39) only, against the same run with the connector off.**

| condition | mean | p50 | max | speedup | fact EM |
|---|---:|---:|---:|---:|---:|
| control (connector off) | 0.725 s | 0.653 | 2.204 | 1.00× | **14/14** |
| `max_churn=0.25` | 2.845 s | 2.520 | 11.282 | **0.25×** | 7/14 |
| `max_churn=1.0` | 0.568 s | 0.263 | 3.519 | **1.27×** | 4/14 |

And the same turns split by kind, which is what the sampling fix bought:

| condition | reuse turns | refresh turns |
|---|---|---|
| `max_churn=0.25` | n=13, mean 0.269 s, **EM 2/8** | n=14, mean 5.238 s, EM 5/6 |
| `max_churn=1.0` | n=22, mean 0.253 s, **EM 1/11** | n=5, mean 1.955 s, EM 3/3 |

**The two halves fail in opposite directions.** A reuse turn is 2.7× faster than the control and answers correctly 25% / 9% of the time. A refresh turn answers correctly 5/6 and 3/3 — indistinguishable from the control — and costs 2.7–7× *more* than the control turn it replaces. Plain append turns are perfect on both axes (0.13 s, EM 6/6). The churn ceiling behaves exactly as a dial should — it moves the reuse fraction from 32% to 55% and EM moves with it, 0.50 → 0.29 — but there is no setting where both are good, because reuse itself is wrong at every churn level observed (`max_churn=0.25` holds churn under 0.248 and still scores 2/8). The best latency on offer is **1.27× at EM 0.29 against a control of 1.00**. That is not a win; it is a 71-point correctness regression bought for 27% latency.

**Why a refresh turn costs more than the full prefill it is equivalent to, and what it is not.** A refresh recomputes the whole window, exactly as the control does, so it should cost what the control costs (0.725 s). It costs 5.2 s. The store is not the cause: pre-sizing worked, and the run logs zero evictions, zero refusals, and loads at 2.8 ms for 665 MB (229 GB/s). The save volume is not the cause either — `saved_token_layers` over the run is one pass per turn, as the CPU harness predicted. I tested one concrete hypothesis on hardware: the save path gathers with `kv[blk, :, off]`, which on the HND layout (this model's) separates two advanced indices with a slice and takes PyTorch's slow path, while loads go through the fused Triton kernel; and the slot transfer and block/offset split were being repeated inside the 40-layer loop. Hoisting both and permuting to make the indices adjacent came back at **6.0 s against 5.2 s — unchanged within noise**. The hypothesis is wrong and the refresh cost is still unexplained. The change is kept because it is strictly less work and is pinned value-identical by a test, but it is labelled in the source as fixing nothing measured.

**Limits.** One run per cell, one model, one window, one paging policy; 14 scored turns per condition. The reuse/refresh EM split is the strongest result here (2/8 and 1/11 against 5/6 and 3/3) and it rests on 14 and 14 scored turns. The refresh-turn cost is measured but not diagnosed; until it is, even "refresh everything" is worse than simply turning the connector off on a paged workload, which is the current recommendation.

@acrosley 2026-08-20
