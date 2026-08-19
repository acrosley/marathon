# Marathon Phase 1 — Position-Shifted KV Reuse for Edited Contexts

**Technical report** · Andrew Crosley · 2026-08-19 · Phase 1 exit criterion met 2026-08-18

Consolidated from the lab notebook in [findings.md](findings.md); every number below cites the dated
entry it came from. Where two entries disagree, the later one is quoted and the correction is stated.
Design context is [DESIGN.md](../DESIGN.md), the roadmap is [PLAN.md](PLAN.md), the wire and connector
contracts are [protocol.md](protocol.md), and the field survey is [related.md](related.md).

---

## 1. Abstract

Provider and engine prefix caching reuses a KV cache only while the prompt is a byte-exact prefix of a
cached one. Any edit to earlier history invalidates everything after it, so the edit turn costs a full
recompute. We measured that collapse on the Anthropic API (cache reads → 0, ~4× TTFT) and on
self-hosted vLLM 0.27.1 (0.12 s → 1.29 s at 12.6k tokens). Phase 1 replaces the all-or-nothing rule
with a delta-driven reuse plan: an rsync-style byte matcher over canonical history lines emits reusable
segments, each carrying its own token shift δ; unchanged spans are reused with their cached values
verbatim and their cached keys re-rotated by δ — exact, because RoPE is a rotation — and only the
edited spans plus the new input are prefilled. Implemented as a vLLM KV connector with a session-keyed
store and a fused Triton scatter-and-re-rotate kernel (1.43 TB/s), the edit turn on Qwen3-14B-FP8 goes
from 0.42→3.94 s (prefix caching, linear in history) to 0.15→0.26 s over 4k–30k tokens: **15.1× at
30k**, with byte-identical output. Across 144 sessions on Qwen3-8B, reuse forwards 1.5% of tokens for
median KL 0.003 versus full recompute; the single failure class is edits to *governing* spans.

---

## 2. Problem

Prefix caching is a prefix match. KV entries encode absolute position and attention is not
position-independent, so a cache entry is valid only if every token before it is unchanged *and* still
at the same index. One edited character anywhere in the history invalidates the entire suffix. This is
not an implementation limit of any one vendor; it is the shape of the mechanism.

**Measured on the Anthropic API** (`claude-haiku-4-5`, `marathon.live_probe`). With append-only
history, canonical serialization hits the cache exactly — past the 4k-token cache minimum, unchanged
turns bill 3 uncached tokens and read the whole history from cache
(*Live probe: append-only history hits the prompt cache*, 2026-08-18).

| turn | ttft_s | input | cache_read | cache_creation | note |
|---:|---:|---:|---:|---:|---|
| 9 | 0.83 | 3 | 0 | 4264 | first cache write |
| 10 | 1.24 | 3 | 4264 | 425 | reads all prior history |
| 11 | 2.47 | 3 | 4689 | 425 | |

A one-word edit to the first message, injected at turn 13, collapses it
(*one early edit collapses the cache*, 2026-08-18):

| turn | ttft_s | cache_read | cache_creation | note |
|---:|---:|---:|---:|---|
| 12 | 0.54 | 5114 | 425 | steady state |
| 13 | 1.95 | **0** | 5968 | edit: full collapse, re-billed at 1.25× write cost |
| 14 | 0.99 | 5968 | 425 | rebuilt |

**Measured locally** (`Qwen/Qwen3-14B-FP8`, vLLM 0.27.1, RTX 5090 / WSL2), without network noise
(*Phase 1 stack up*, 2026-08-18):

| turn | prompt_tokens | prefill_s | prefix_hit | Marathon wire_bytes |
|---:|---:|---:|---:|---:|
| 19 | 12,027 | 0.117 | 11,408 | 3,361 |
| 20 | 12,632 | **1.285** | **0** | 4,524 |
| 21 | 13,233 | 0.122 | 12,624 | 3,396 |

The contrast that motivates the project is on that one row: the serving layer throws away and
re-processes ~12.6k tokens while Marathon's own delta absorbs the identical edit in +1.2 KB of wire.
The delta engine already knows the edit is cheap; before Phase 1 the serving side could not use that.

---

## 3. Method

### 3.1 Delta engine → reuse plan

`marathon.diff` is an rsync-style rolling-block matcher; `marathon.canonical` guarantees that logically
identical history serializes to identical bytes and that `serialize_history(h[:k])` is a byte-prefix of
`serialize_history(h[:k+1])`. `reuse_plan.plan(old, new, tokenize, head_tokens=…)` runs the matcher over
**canonical JSONL lines** — the ledger's own unit — so a match is always a whole run of history entries
and never needs snapping to a token boundary.

The plan is a list of `Segment(src_start, src_end, dst_start)` in destination order, each with its own
`delta = dst_start − src_start`. The classical single-span vocabulary is a derived view of it: `P` the
leading prefix, `E'` the first recomputed span, `S` the last reused segment. Everything the segments do
not cover is recomputed. The plan also carries:

| field | meaning |
|---|---|
| `moved` | per segment: did its entries *relocate* (index changed) rather than merely shift? Relocated runs are recomputed by default |
| `policy` | `reuse` / `repair` (recompute the first `repair_first` tokens of each non-leading segment) / `full` (nothing survived) |
| `repair_first` | default 256, 0 unless policy is `repair` |
| `total` | length of the new sequence in the serving layer's token coordinates |

A message is **governing** when it serializes `"governing": true`; the `system` role gets it by default,
and the key is written only when true, so canonical bytes of sessions that never set it are unchanged.
A governing edit trips `repair`. Two subtleties are load-bearing and were found by measurement, not
design: duplicate lines are matched to the **nearest** candidate, not the first (byte identity is not
context identity), and matching is **order-agnostic**, so a segment's source range may run backwards and
its δ may be negative.

### 3.2 Position-shifted KV reuse via exact RoPE re-rotation

A key computed at position `p` and needed at position `p+δ` requires exactly one further rotation:

> `K(p+δ) = R(δ·θ) · K(p)`, where `R` is RoPE's per-pair rotation and `θ_i` the model's own `inv_freq`.

Because RoPE is a rotation applied *after* the key projection, this is exact, not an approximation.
Measured against the real models' `inv_freq`: max abs error **1.3e-05** (Qwen3-0.6B, fp32),
**1.39e-05** (Qwen3-8B), and **8e-6 – 1.5e-5** on Qwen3.5-4B through the model's own rotary module
under partial rotary + mRoPE. `tests/test_kvshift.py` proves the identity in CI for
δ ∈ {−7, −1, 0, 1, 13, 64}.

**V needs nothing.** RoPE is applied only to queries and keys; values are never position-encoded, so a
cached V is valid at any position it is placed. The entire positional correction is one rotation of the
K half.

What re-rotation does *not* fix is that a reused span's KV attended to the *old* text before it. That
residual — stale attention, not stale position — is the whole quality question of Section 4d, and it is
also why a *relocated* block (whose entire preceding context changed) is recomputed rather than
transplanted.

### 3.3 vLLM connector

`marathon.vllm_shift_connector.MarathonShiftConnector` is a `KVConnectorBase_V1` (loaded via
`kv_connector_module_path`, `kv_role=kv_both`). Requests carry
`{"session": id, "load": [...], "save": true}` in `SamplingParams.extra_args["kv_transfer_params"]`.

- **SAVE** — after each layer is written, gather the KV of the positions this scheduler step actually
  computed (`num_computed_tokens` onward) out of vLLM's paged cache into a per-session, per-layer,
  position-indexed buffer. A save at a lower `dst_start` truncates positions above it, which is exactly
  what an edit means.
- **LOAD** — `get_num_new_matched_tokens` reports `dst_end − num_computed_tokens` as externally
  available so vLLM skips prefilling that span; `start_load_kv` copies it in, V verbatim and K
  re-rotated by δ, straight into the paged layout.
- **The phase trick.** vLLM's connector API can express externally-matched tokens only as a *prefix*.
  `k` reused segments are therefore handed over as `k + 1` requests (`local_probe._phases`), each
  stopping on the block boundary where the next segment begins; the final request is the real one.
  This is the probe's job, not the connector's — a production server would take the span from the delta
  engine directly.
- **Store** (`marathon.shift_store`) — keyed by session id, one contiguous slab per session per layer
  (`SLAB = 16384` positions, doubling only if outgrown), a total token budget
  (`MARATHON_STORE_TOKENS`, default 32768) with LRU eviction of whole sessions, and one in-flight
  writer per session *enforced*. A request with no session id is pass-through. Because eviction happens
  on the worker while the scheduler is the side that promises vLLM a span needs no prefill, the
  scheduler mirrors the same bookkeeping tensor-free and **declines** a load whose source positions were
  evicted — a miss is a recompute, never a wrong answer.
- **Kernel** (`marathon.shift_kernels`) — one fused Triton pass per layer: read each source token's
  `[K|V]` row, rotate K in-register in fp32, write it into its destination block slot. Branch-free via
  two precomputed tables, `out[i] = k[i]·cos[i] + k[partner[i]]·sgn[i]`, matching
  `kvshift.rerotate_keys` arithmetic in the same order. Both paged layouts (HND/NHD) are handled by
  passing strides; a torch fallback is selected when Triton or CUDA is unavailable.

---

## 4. Results

### (a) North-star: edit-turn TTFT is near-flat as the session grows

`scripts/phase1_lengthsweep.sh`, Qwen3-14B-FP8. `--turns` is chosen so the edit turn (always
last-but-3) lands near each target size; `steady` is the mean prefill of the three turns before the
edit (*North-star*, 2026-08-19).

| edit-turn prompt tok | turns | prefix edit_s | shift edit_s | speedup | steady_s prefix/shift | reused tokens | copy ms (MB, GB/s) |
|---:|---:|---:|---:|---:|---|---:|---|
| 4,206 | 10 | 0.4193 | 0.1542 | 2.7× | 0.072/0.074 | 2,976 | 24.4 (465, 18.6) |
| 8,382 | 17 | 0.6936 | 0.1544 | 4.5× | 0.082/0.080 | 7,152 | 16.0 (1118, 68.4) |
| 12,561 | 24 | 1.2723 | 0.1951 | 6.5× | 0.100/0.101 | 11,328 | 30.2 (1770, 57.2) |
| 16,143 | 30 | 1.7840 | 0.2006 | 8.9× | 0.113/0.107 | 14,912 | 35.5 (2330, 64.1) |
| 24,501 | 44 | 2.9281 | 0.3612 | 8.1× | 0.139/0.141 | 23,264 | 158.4 (3635, 22.4) |
| 30,471 | 54 | 3.8688 | 0.3663 | 10.6× | 0.156/0.162 | 29,232 | 151.6 (4568, 29.4) |

Mid-history edits (edited message in the middle, not turn 0) — prefix caching keeps the first half as a
hit, so it starts lower, but the *shape* is identical:

| edit-turn prompt tok | prefix edit_s | shift edit_s | speedup |
|---:|---:|---:|---:|
| 16,143 | 0.9550 | 0.2224 | 4.3× |
| 30,471 | 2.5715 | 0.3579 | 7.2× |

Prefix caching's edit turn is a straight line in context length: **131 µs per token of history**. Shift
mode's slope is **8.1 µs/token — 16× shallower**. Steady-state prefill is untouched and identical
between modes (0.072 → 0.162 s over the range), which was the point of leaving prefix caching enabled
underneath. Every row's parity answer is `7391-KAPPA` and every shift row's text matches its prefix row
on every turn.

**Honest qualifier:** shift TTFT is not literally flat. The re-rotate-and-scatter copy is linear in the
reused span and, in that run, accounted for essentially the whole 8.1 µs/token slope (41% of the 0.37 s
edit turn at 30k). The claim is *sub-linear by 16×, with a memory-bandwidth term linear in |S|* — not
O(1). Section 4f shows what happened when that term was attacked.

### (b) Single, mid, and grow edits on 14B

First serving-path result (*Position-shifted KV reuse inside vLLM*, 2026-08-18), 12.5k-token session,
edit of turn 0 on turn 20:

| turn | prompt_tokens | prefix_s | shift_s | shift prefix_hit | text |
|---:|---:|---:|---:|---:|---|
| 19 | 11,960 | 0.132 | 0.128 | 11,344 | `<think>` |
| 20 | 12,561 | **1.465** | **0.242** | 640 | `<think>` |
| 21 | 13,158 | 0.135 | 0.138 | 12,544 | `<think>` |
| 23 | 14,364 | 0.236 | 0.236 | 13,744 | `7391-KAPPA` |

**6.1×.** The connector logged `loaded 11328 tokens x 40 layers from store[620:11948], delta=4` — 90.2%
of the prompt copied-and-re-rotated, 9.8% forwarded, matching the HF prototype's predictor. Turn 23
reads the planted fact *through* turn 20's re-rotated KV.

Position and size of the edit (*mid-history edit and grow edit*, 2026-08-18):

| variant | prefix edit_s | shift edit_s | speedup | note |
|---|---:|---:|---:|---|
| edit of turn 0 (δ=+4) | 1.465 | 0.242 | 6.1× | prefix fully collapsed |
| mid-history (`--edit-turn 10`, δ=+4) | 0.950 | 0.200 | 4.7× | prefix gets a partial hit on `P` |
| grow (`--edit-grow 200`, δ=+186) | 1.273 | 0.214 | 5.9× | 46× larger shift, same cost |

Speedup is indifferent to |δ|, as expected — re-rotation is a fixed-cost angle. The same session after
the session-keyed store rewrite came out at **1.2354 → 0.1629 s, 7.6×**, the fastest number this
workload has produced (*The shift connector becomes session-keyed and scheduler-safe*, 2026-08-19).

The grow-edit run flipped one greedy token on turns 20–21 (`'1'` vs `'<think>'`), initially read as a
quality signal. A control overturned that reading (*`--repair-first` does not fix the grow-edit token
flip*, 2026-08-18): `--repair-first M` at M ∈ {0, 64, 256, 1024} never moved the token, and M=16384 —
which clamps to reusing 16 tokens, i.e. a full recompute in all but name at 1.243 s — emitted a *third*
answer, `' ok'`. Three phase structures over byte-identical token ids give three different first tokens.
The flip is chunking noise at the numerical-noise level, not re-rotated KV.

### (c) Multi-span cost and moved blocks

Before this work `reuse_plan` returned `policy="full"` on seeing a second edited span. Removing that
ceiling (*Multi-span and moved-block KV reuse*, 2026-08-19), Qwen3-14B-FP8, 12.5k-token prompt:

| config | turn19 | turn20 | turn21 | requests | segments | vs prefix |
|---|---:|---:|---:|---:|---:|---:|
| prefix (k=1 mutation) | 0.097 | 1.243 | 0.107 | 1 | – | – |
| prefix (pure move) | 0.089 | 1.181 | 0.109 | 1 | – | – |
| shift `--edit-count 1` | 0.097 | 0.198 | 0.105 | 2 | 2 | **6.3×** |
| shift `--edit-count 2` | 0.099 | 0.276 | 0.107 | 3 | 3 | 4.5× |
| shift `--edit-count 4` | 0.097 | 0.411 | 0.105 | 5 | 5 | 3.0× |
| shift `--edit-count 8` | 0.090 | 0.664 | 0.100 | 9 | 9 | 1.9× |
| shift `--move` | 0.098 | 0.293 | 0.110 | 3 | 5 | 4.0× |
| shift `--edit-count 4 --move` | 0.095 | 0.546 | 0.106 | 6 | 9 | 2.3× |

Every row's generated text is byte-identical to prefix mode on every turn. The edit turn is almost
exactly linear in k: **+66 ms per additional edited message**, which is the k+1-request phase trick plus
that message's own prefill — nothing that scales with context. Extrapolating, shift stops beating the
flat 1.24 s collapse at about **k ≈ 17 edited messages** in a 12.5k context.

Quality at k, Qwen3-8B, worst case over each scenario's fact questions:

| scenario | segments | policy | frac | klmean | klmax | tf_top1 | QA | ==ref |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| multi-k1 | 3 | reuse-all | 0.012 | 0.0014 | 0.0128 | 1.00 | 2/2 | 2/2 |
| multi-k2 | 6 | reuse-all | 0.018 | 0.0012 | 0.0111 | 1.00 | 3/3 | 3/3 |
| multi-k4 | 8 | reuse-all | 0.028 | 0.0008 | 0.0095 | 1.00 | 4/4 | 4/4 |
| multi-k8 | 22 | reuse-all | 0.049 | 0.0022 | 0.0259 | 1.00 | 4/4 | 4/4 |
| move | 5 | reuse-all | 0.009 | 0.0005 | 0.0033 | 1.00 | 2/2 | 2/2 |
| combined | – | reuse-all | 0.028 | 0.0007 | 0.0049 | 1.00 | 3/3 | 3/3 |
| combined | – | **no-rerotate** | 0.028 | **2.3173** | **13.5002** | **0.58** | **1/3** | 1/3 |

At multi-span the no-rerotate control stops being a KL penalty and becomes a correctness failure.
Re-rotation is not a refinement; it is what makes non-prefix reuse work at all.

**A relocated block is not a shifted block.** Transplanting relocated blocks the same way as shifted
ones ran fine mechanically (`delta=-10153` and its `+10154` mirror, 0.235 s, no declines) and produced
garbage: turn 20 emitted `'1'`, turn 21 `'2'`, and turn 23 answered
`' content used to grow the context in a 2 2 2 2'` instead of `7391-KAPPA`. `--repair-first 256` did not
move it, ruling out a seam effect. A block that moved 10k positions has KV summarising a *completely
different* prefix; re-rotation fixes where it now sits, not what it attended to. `reuse_plan` now flags
relocations in `plan.moved` and recomputes them (2 × ~600 tokens of prefill), which turned the broken
row into the working one: **pure move 1.181 → 0.293 s (4.0×) with byte-identical output.**

The same hazard one level down is **byte identity is not context identity**: `multi-k1` has no move at
all, yet its deltas are `[-2385, 0, 4]` — repeated filler prose matched a copy of itself 2,385 tokens
earlier. In the serving path this bit: `_match` took the first unused byte-identical entry, mapping a
bare `"ok"` acknowledgement to one 10k tokens away. It now takes the nearest.

### (d) Quality

**HF prototype, Qwen3-0.6B** (*re-rotated shifted KV holds quality at 0.7% recompute*, 2026-08-18).
`klmean` is mean KL vs full recompute over a teacher-forced continuation; `frac` is tokens forwarded;
`eff` adds the blend policy's layer-0/1 scan over all of `S`:

| scenario | policy | frac | eff | klmean(worst) | tf_top1 | QA | prefill_s |
|---|---|---:|---:|---:|---:|---:|---:|
| edit-turn0 (δ=+4) | full-recompute | 1.000 | 1.000 | 0.0000 | 1.00 | 3/3 | 0.058 |
| | no-rerotate | 0.007 | 0.007 | 0.0168 | 1.00 | 3/3 | 0.024 |
| | reuse-all | 0.007 | 0.007 | 0.0033 | 1.00 | 3/3 | 0.023 |
| | first-128 | 0.031 | 0.031 | 0.0036 | 1.00 | 3/3 | 0.024 |
| | blend-r0.15 | 0.155 | 0.226 | 0.0022 | 1.00 | 3/3 | 0.035 |
| edit-mid (δ=+4) | reuse-all | 0.007 | 0.007 | 0.0027 | 0.98 | 3/3 | 0.023 |
| | first-128 | 0.032 | 0.032 | 0.0009 | 0.98 | 3/3 | 0.024 |
| edit-grow (δ=+209) | no-rerotate | 0.089 | 0.089 | 0.0295 | 0.92 | 3/3 | 0.025 |
| | reuse-all | 0.088 | 0.088 | 0.0024 | 1.00 | 3/3 | 0.026 |
| | blend-r0.30 | 0.219 | 0.291 | 0.0017 | 1.00 | 3/3 | 0.039 |

**Qwen3-8B, dependent-suffix scenarios** (*instruction spans, not fact spans*, 2026-08-18), where `S`
genuinely depends on the edited span:

| scenario | policy | frac | klmean | kl_first | tf_top1 | answer | ==ref |
|---|---|---:|---:|---:|---:|---|---:|
| dep-anaphora (`S` → `E'` reference) | reuse-all | 0.005 | 0.0145 | 0.0199 | 0.94 | new value | 2/3 |
| dep-contradict (override) | reuse-all | 0.014 | 0.0122 | 0.0579 | 0.96 | honours override | 1/2 |
| dep-instruction (governing) | reuse-all | 0.004 | 0.0118 | **0.3492** | 0.95 | de,de | 0/2 |
| dep-instruction | first-32 | 0.010 | 0.0036 | 0.0384 | 0.98 | de,en | 1/2 |
| dep-instruction | first-512 | 0.100 | 0.0014 | 0.0177 | 0.98 | de,en | 1/2 |
| dep-instruction | blend-r0.30 | 0.300 | 0.0051 | 0.1761 | 0.95 | de,de | 0/2 |

Fact-level dependence does not break reuse: the query attends to `E'` directly, so a fact only has to
survive in `E'` while `S`'s stale states carry the *pointer*, which the edit did not change. At 8B,
prefill is 0.42–0.53 s full vs **0.04 s** reused (~11×) — compute-bound enough that recompute fraction
and wall clock agree.

**Distribution eval, 144 sessions.** Two runs of `marathon.kvshift_eval` on Qwen3-8B: 60 sessions /
200 items at seed 1234 and 84 sessions / 269 items at seed 1235. Sessions are 4.4k–8.2k tokens across
three families (coding on this repo's source, prose from its docs, Q&A over seeded fact tables), each
rendered through Qwen3's chat template, each given one edit of a known kind, with three codes planted
before / inside / after the edit. `prefix-equiv` is what vLLM prefix caching can do — reuse `P` only —
and is here for its *cost*.

Seed 1234, overall (*Distribution eval, 60 sessions*, 2026-08-18):

| condition | n | klmean | klmed | klp95 | klmax | tf_top1 | exact | frac | >.05 | >.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-recompute | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 1.00 | 1.000 | 0 | 0 |
| reuse-all | 200 | 0.0142 | 0.0035 | 0.0599 | 0.6382 | 0.985 | 0.64 | **0.016** | 11 | 1 |
| no-rerotate | 200 | 0.0675 | 0.0128 | 0.3021 | 1.5246 | 0.969 | 0.45 | 0.016 | 53 | 18 |
| prefix-equiv | 200 | 0.0017 | 0.0011 | 0.0038 | 0.0452 | 0.992 | 0.80 | **0.664** | 0 | 0 |

By edit kind (seed 1234), showing where re-rotation earns its keep:

| edit kind | n | klmean (reuse) | klmean (no-rr) | tf_top1 | exact | frac |
|---|---:|---:|---:|---:|---:|---:|
| fact (δ≈0) | 41 | 0.0023 | 0.0025 | 0.990 | 0.76 | 0.005 |
| insert (δ≈+29) | 41 | 0.0067 | 0.0182 | 0.988 | 0.66 | 0.009 |
| delete (δ≈−14) | 39 | 0.0115 | **0.1161** | 0.985 | 0.59 | 0.004 |
| rewrite (δ −135…+137) | 41 | 0.0167 | **0.1637** | 0.985 | 0.71 | 0.058 |
| governing (δ≈0–1) | 38 | 0.0350 | 0.0372 | 0.975 | 0.45 | 0.004 |

Re-rotation helps exactly where δ moves and is a no-op where δ ∈ {−1,0,1} — predicted behaviour of a
fixed-angle rotation, measured over 200 items rather than asserted.

**The governing 2×2.** Seed 1234 argued that "governing" was merely a proxy for "at position 0, so `S`
is the whole history" — the system prompt and the front position being perfectly confounded. Seed 1235
added two kinds to break the confound (`early-fact`: front, non-governing; `mid-governing`: a standing
instruction moved into a mid-history user turn) and **overturned that reinterpretation**
(*The 2×2 settles it*, 2026-08-18):

| cell | n | klmean | klmed | klp95 | klmax | exact | mean \|S\| | >.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front, governing | 40 | 0.0264 | 0.0081 | 0.0601 | 0.3676 | 0.57 | 6,060 | 4 |
| front, non-governing | 39 | **0.0029** | 0.0018 | 0.0100 | 0.0132 | 0.87 | 5,652 | **0** |
| mid, governing | 38 | 0.0255 | 0.0040 | 0.1134 | 0.5133 | 0.68 | 3,533 | 3 |
| mid, non-governing | 39 | **0.0020** | 0.0012 | 0.0058 | 0.0110 | 0.67 | 3,452 | **0** |

The governing flag moves mean KL by ~**9×**; position moves it by nothing. `early-fact` puts 5,652
tokens of stale-attention `S` after the edit — as much as the governing case — and is the *cleanest*
cell in the run. Conditioned on the flag, among 191 non-governing items **p95 KL never crosses 0.05 in
any |S| bin out to 8k tokens** or any downstream-fraction bin, and spearman(KL, |S|) = −0.162; among 78
governing items it crosses in 3 of 6 bins with no ordering. Over all 269 items
spearman(KL, |S|) = −0.028 and spearman(KL, |S|/prompt) = +0.042 — no relationship — while
spearman(KL, |δ|) = +0.237. The worst item in the run (`sid=69`, mid-governing) has |S| = 1,678 and a
downstream fraction of 0.23: a size threshold would have missed it and needlessly refused hundreds of
large-|S| fact edits that are fine.

**Conclusion for `reuse_plan`: keep the governing flag, add no position or |S| threshold.** The flag is
conservative — 71/78 governing items are under KL 0.05 and would have been fine — so `repair` rather
than `full` remains the right response, which is what `reuse_plan` already emits.

Seed reproducibility: the five original kinds rank identically at seed 1235 (fact 0.0020, insert 0.0055,
delete 0.0062, rewrite 0.0141, governing 0.0264), overall reuse-all 0.0119 / med 0.0030 / p95 0.0394 at
1.5% forwarded, `prefix-equiv` 68.2% forwarded for KL 0.0014 — a **45× cost ratio for an 8.5× KL
difference**. The no-rerotate control is again 6.8× worse.

**Metric calibration a reviewer should hold onto:** `prefix-equiv` scores KL 0.0017 and exact-match
**0.80**, not 1.00, against a single-pass reference over byte-identical token ids. Free-running 32-token
exact match has a ~20% noise floor on this workload, so reuse-all's 0.64 reads against 0.80. The
teacher-forced KL has no such floor and separates the conditions cleanly (0.0017 / 0.0142 / 0.0675) —
which is why the KL columns carry the argument and exact-match does not.

**Selective recompute does not pay.** Consistently, across three independent measurements:

| where | finding |
|---|---|
| HF prototype (0.6B) | `first-M` and blend-r cut KL 2–4× from an already negligible base; the blend selector's layer-0/1 scan over all of `S` costs more (`eff` − `frac` ≈ 0.07) than the recompute it selects |
| vLLM (14B) | `--repair-first` M=0→1024 adds 72 ms (+34% on the edit turn) and changes no output; M=64 already costs 26 ms, mostly per-request scheduling |
| multi-span (8B) | `first-32/128/512` and `blend-r0.05/0.15/0.30` leave klmean in the same 0.001–0.010 band for 2–16× the compute (`first-512` reaches 80% recompute at k=8) |

The one place selective recompute *does* buy something is a governing edit (first-token KL 0.35 → 0.038
at first-32), and even there it restores agreement in only half the cases.

### (e) CacheBlend / LMCache — the negative result

Getting a fair CacheBlend number took four entries of engineering (*CacheBlend runs end to end*,
*LMCache built from source*, *Fully native CacheBlend*, *CacheBlend knobs*, 2026-08-18). The PyPI
`lmcache==0.5.3` wheel's `c_ops` is ABI-incompatible with torch 2.13, and its pure-torch fallback has no
branch for vLLM 0.27's fused KV layout. No wheel exists for torch 2.13 and WSL has neither nvcc nor
root — but torch's cu13 wheels *bundle* nvcc 13.3.73, so `scripts/phase1_build_lmcache.sh` builds
LMCache from source in 53 s. A further 38-line CUDA patch
(`scripts/lmcache_fused_single_layer.patch`, unit-checked bit-exact, worth upstreaming) adds the fused
formats to `single_layer_kv_transfer`. Only then is the measurement honest:

| turn | prompt_tokens | blend (native) | none | prefix |
|---:|---:|---:|---:|---:|
| 19 | 11,947 | 0.845 | 1.367 | 0.117 |
| 20 (edit) | 12,548 | **1.260** | **1.265** | **1.285** |
| 21 | 13,145 | 0.981 | 1.425 | 0.122 |

**On the edit turn CacheBlend-as-shipped is a tie with full recompute, and it is ~7× worse than prefix
caching on every unchanged turn** (prefix caching is off in blend mode). LMCache retrieves ~99.7% of
tokens every turn and recomputes a fixed 15%, so the reuse is real — the cost is where it lands. Per-layer
DEBUG timestamps on an 11.3k-token turn: `LMCBlender.blend` 804 ms of 977 ms (82%), of which layers
2–39 over the selected 15% is 586 ms. The KV move itself, measured directly at turn-20 scale (2.06 GB,
40 layers): **77 ms, 26.7 GB/s — memcpy speed.** So transfer is ~2 ms of the 15 ms per layer; ~90% of
blend time is LMCache's eager Python Qwen3 recompute at ~9 µs/token/layer against vLLM's ~2.5 µs —
**3.6× less efficient per token than the prefill it replaces.** 0.15 × 3.6 ≈ 0.55, plus two full-length
passes and the vLLM tail, lands blend on the recompute-everything line.

**The recompute ratio is the wrong knob:**

| turn | prompt tok | none | prefix | blend r=0.15 | blend r=0.05 | blend r=0.02 |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 11,960 | 1.691 | 0.098 | 0.789 | 0.580 | 0.521 |
| 20 (edit) | 12,561 | 1.718 | 1.377 | **1.399** | **1.395** | **1.369** |
| 21 | 13,158 | 1.685 | 0.116 | 0.878 | 0.614 | 0.507 |

0.15 → 0.02 buys ~1.7× on *unchanged* turns and **nothing** on the edit turn, because the edit turn's
cost is the two full-length passes plus vLLM's prefill of the new tokens, not the selected fraction.
Every mode and ratio answered the planted fact `7391-KAPPA` exactly, so this is a cost result, not a
quality one.

The escape hatch — blend plus vLLM prefix caching, so unchanged turns keep the 0.12 s path — is closed
by an LMCache bug: with prefix caching on, the scheduler hands LMCache only the new tokens after a hit
while the blender still assumes the full chunked prompt, and it dies on the first hit
(`RuntimeError: The size of tensor a (1208) must match the size of tensor b (3)` in
`blender.process_qkv`), after which the engine hangs until the pin monitor times out at 300 s.

**The idea is not refuted; this implementation of it is.** With 15% recomputed at vLLM's efficiency the
edit turn would be ~0.25 s, a ~5× win — which is roughly what position-shifted reuse actually delivers,
by a different route that recomputes nothing on shifted-but-unchanged spans.

### (f) Systems

**Connector safety** (*session-keyed and scheduler-safe*, 2026-08-19). Two interleaved sessions,
Qwen3-0.6B, each planting a different access code and each editing turn 0 on turn 9:

| turn | session | prefix_s | shift_s | reused | parity (both modes) |
|---:|---|---:|---:|---|---|
| 9 | s0 | 0.0578 | 0.0258 | 4800/6034 | |
| 9 | s1 | 0.0595 | 0.0249 | 4800/6034 | |
| 11 | s0 | 0.0291 | 0.0289 | – | `7391-KAPPA` |
| 11 | s1 | 0.0289 | 0.0285 | – | `5820-OMEGA` |

All 24 rows are byte-identical between modes and each session answers **its own** code. Store ends at
`{'s0': 5414, 's1': 5414}` with `hits: 2, misses: 0, evictions: 0, refusals: 0`. Making the connector
correct also made it faster: the 14B regression check came out at **1.2354 → 0.1629 s, 7.6×**, against
0.242 s for the original probe.

Two bugs only a run could find. (1) The first rewrite required a session's positions to be contiguous
from 0 and refused any save leaving a hole — so *every* save was refused (`refusals: 9`, load declined,
silent fall back to full recompute, parity still correct and speedup gone), because a request only ever
computes what vLLM's prefix cache did not already have. The store now records a `base` as well as a
high-water mark. (2) Growing buffers in 1024-token steps reallocated all 40 layer tensors every other
turn: the copy fell to **4.7 GB/s (364 ms)** and every turn slowed 2–4×, including turns that run no
connector code — the tell that it was allocator churn.

**Store cost.** 164 KB/token on Qwen3-14B (40 layers × 8 KV heads × 128 dim × 2 × bf16). 2.7 GB for a
16k buffer, **5.45 GB for 32k**. The 24k and 30k sweep points needed `gpu_memory_utilization 0.78`
instead of 0.80, peaking at 27.1 GB of 32.6 GB.

**Kernel** (*A fused Triton kernel takes the shifted copy to memcpy speed*, 2026-08-19). In torch the
copy is five passes over the same bytes (slice K, upcast, materialise `rotate_half`, fuse, downcast,
`cat`, scatter). Triton makes it one. Micro-benchmark, Qwen3-14B shapes, δ=186, best of 5; GB/s counts
source read plus destination write:

| tokens | MB | torch ms | GB/s | µs/tok | triton ms | GB/s | µs/tok | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 640 | 6.52 | 191.6 | 1.593 | 1.01 | 1233.5 | 0.247 | 6.4× |
| 12,288 | 1,920 | 20.96 | 178.9 | 1.706 | 2.80 | 1339.3 | 0.228 | 7.5× |
| 30,720 | 4,800 | 61.48 | 152.5 | 2.001 | 6.54 | **1433.0** | 0.213 | **9.4×** |

1,433 GB/s is ~95% of the 5090's ~1.5 TB/s device-to-device peak; a replication at real connector sizes
(40 layers of 56k-token paged cache, 5.4 GB store, scattered block table) gives the same number, so it
is not a small-tensor artefact. The 30k sweep point re-measured:

| | before (torch copy) | after (Triton copy) |
|---|---:|---:|
| edit turn, shift | 0.3663 s | **0.2603 s** |
| of which copy_ms | 151.6 ms | 34.4 ms |
| edit turn, prefix | 3.8688 s | 3.9391 s |
| speedup | 10.6× | **15.1×** |
| steady-state prefill (prefix/shift) | 0.156/0.162 s | 0.1583/0.1633 s |

Copy is 4.4× cheaper and now 13% of the edit turn instead of 41%; the edit turn is 1.41× faster and the
gap at 30k widens to **15.1×**. All 54 turns' text is identical between modes. Of the remaining 34.4 ms
the kernel itself is 6.2 ms; the rest is one-shot cost (40 store reads, slot transfer, first-touch of
buffers written by the preceding turn) paid once per edit turn, not per token. **87% of the 0.26 s edit
turn is now vLLM prefilling the edited span and the new tokens — the term the design says has to be
there.**

Two traps worth recording, because both would have been misread as "the kernel is slow": the first
in-engine measurement came back at 216 ms because an edit turn issues exactly one load, so Triton's JIT
compile was charged in full to the one thing it was meant to accelerate. Warming at
`register_kv_caches` fixed it only to 74 ms, because Triton specialises on argument divisibility and a
1-token warmup compiles a different kernel from the 29,232-token load that follows. Warming with both a
16-divisible and a ragged token count got the real 34.4 ms.

**Where the earlier bandwidth numbers went.** The sweep's 22–68 GB/s figures were measuring
fragmentation of a store allocated in pieces, not memory bandwidth. Allocating a session's buffers once
as a slab took the same 1,770 MB copy from 104 ms to **3.15 ms (549 GB/s)** before Triton was involved
at all. That does not overturn the sweep's *shape* argument — the copy is still linear in |S| — but it
moves the constant by more than an order of magnitude, and the sweep's "copy is 41% of the edit turn at
30k" is superseded by the 13% above.

### (g) Phase 1.5 — hybrid (Gated-DeltaNet) models

`Qwen/Qwen3.5-4B` (*Phase 1.5*, 2026-08-19): 32 layers, `full_attention_interval=4` → **24 linear
(GDN) / 8 full attention**, `head_dim=256` with `partial_rotary_factor=0.25` and interleaved mRoPE.
The mRoPE is a no-op for text (all three position rows equal), and re-rotation still holds exactly on
the 8 attention layers — max abs error 8e-6 – 1.5e-5 for δ=37, measured through the model's own rotary
module. `rerotate_keys_partial` rotates the leading 64 dims and leaves the rest; that is the entire
delta to `kvshift.py` on the attention side.

A linear layer has no per-token KV, so reuse becomes *replay*. Per-token cost, from parameter counts on
the loaded model: a full token-forward is 3,620 M; one replayed `S` token across all 24 linear layers is
**558 M for replay-hidden (15.4%)** and **50 M for replay-mix (1.39%)** — replay-mix caching each
layer's own post-conv `(q, k, v, beta, g)`, which *is* the linear layer's KV analogue.

| scenario | policy | flops | prefill_s | vs full | kl fact | kl open | tf_top1 | facts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edit-turn0 (S=5,271) | full-recompute | 1.000 | 0.675 | 1.0× | 0.0000 | 0.0000 | 1.00 | 3/3 |
| | stale-state | 0.008 | 0.189 | 3.6× | 0.0154 | 0.0755 | 0.88 | 3/3 |
| | replay-mix | 0.022 | 0.574 | 1.2× | 0.0395 | 0.0228 | 0.92 | **2/3** |
| | replay-mix+first256 | 0.069 | 0.588 | 1.1× | 0.0141 | 0.0166 | 0.90 | 3/3 |
| | replay-hidden | 0.160 | 0.618 | 1.1× | 0.0343 | 0.0211 | 0.92 | **2/3** |
| | replay-hidden+first256 | 0.200 | 0.633 | 1.1× | 0.0037 | 0.0143 | 0.96 | 3/3 |
| edit-mid (S=2,669) | stale-state | 0.008 | 0.204 | 3.3× | 0.0150 | 0.0451 | 0.90 | 3/3 |
| | **replay-mix** | **0.015** | 0.420 | 1.6× | **0.0017** | 0.0139 | 0.92 | 3/3 |
| | replay-hidden | 0.085 | 0.476 | 1.4× | 0.0016 | 0.0103 | 0.92 | 3/3 |
| edit-grow (δ=+701) | no-rerotate | 0.162 | 0.303 | 2.6× | 0.0231 | 0.0461 | 0.96 | 3/3 |
| | stale-state | 0.162 | 0.312 | 2.6× | 0.0175 | 0.0296 | 0.96 | 3/3 |
| | **replay-mix** | 0.168 | 0.511 | 1.6× | **0.0020** | 0.0101 | 0.96 | 3/3 |

**The compute win transfers.** `replay-mix` reaches klmean 0.0017 on facts at **1.5% of full-model
FLOPs** — against the dense eval's 1.5–1.6% of tokens forwarded for median KL 0.0035. Re-rotation still
does real work on the 8 attention layers: on `edit-grow` (δ=+701) the no-rerotate control is worse than
`stale-state` at identical cost (0.0231 vs 0.0175 facts), and roughly ties it where δ=+4.

**What it costs that a dense model does not: memory.** The 8 attention layers' KV over 5.3k tokens is
167 MiB (32 KB/token); the linear layers' mix cache over the same span is 2,988 MiB
(**566 KB/token**) — roughly **18× the cache bytes per token**. `replay-hidden` is 5× cheaper in memory
and 6× more expensive in FLOPs; `stale-state` is nearly free (0.8%) but 5–25× worse in KL.

**A new failure mode dense models do not have.** On `edit-turn0` — the edit 41 tokens in, with 5,271
tokens of `S` after it — both replay policies answered the *pre-edit* code while `stale-state` got it
right. That inversion is the recurrence decaying: replay rolls `E'` into the state and then washes it
out under 5,271 replayed tokens, whereas `stale-state` appends `E'` last, so recency saves it by
accident. This is the mirror image of the dense finding, where `--repair-first` did not help — here
first-M is exactly the repair that works, and it is cheap (`replay-mix+first256`: 6.9% of FLOPs,
0.0395 → 0.0141). The rule this points at is about the *ratio* of |S| to the state's effective memory,
and one scenario cannot calibrate M.

**Wall clock is the prototype's, not the method's.** `replay-mix` is only 1.2–1.6× faster than full
recompute despite doing 1.5–2.2% of the weight FLOPs, because the replay runs transformers'
`torch_chunk_gated_delta_rule` — a Python loop over 64-token chunks in fp32, ~40 chunks × 24 layers per
edit turn, no fused kernel (`flash-linear-attention` is not installed). `stale-state`, which does no
scan, gets the 3.4× its FLOP count predicts. This is the same gap the dense work closed by moving
`kvshift` into vLLM and then onto a Triton kernel.

---

## 5. Limitations and honest caveats

Collected from every entry, unedited in substance.

**Scope of the systems result**

- Single GPU throughout. **Tensor parallelism is untested** — each worker would keep its own store and
  nothing coordinates them.
- **Preemption is untested.** A request preempted and recomputed mid-flight keeps its store positions,
  and nothing has measured whether the re-issue lands on the same coordinates.
- **Chunked prefill interleaved with a load** on the same request is untested; the probe only ever hands
  a load to a request whose local prefix hit already ends at `dst_start`.
- The scheduler/worker bookkeeping mirror stays in step because both sides see the same saves in the
  same order — true for the probe's sequential requests, never stress-tested against a scheduler that
  retries `get_num_new_matched_tokens` or reorders requests. Under real concurrency the two LRU orders
  could drift; the failure mode is a declined load, not a wrong one.
- **Eviction under pressure is covered only by CPU tests**, never by a GPU run that actually evicted a
  live session.
- Reuse is **whole-block only**, so a ragged head or tail of a reused run is recomputed.
- The store is GPU-resident with no CPU/disk tier, so the budget is a hard ceiling, not a paging policy.
  The slab wastes memory on short sessions (a 4k session still takes a 16k slab), and two 14B sessions
  at the default budget are 5.4 GB.
- The two-phase (`k+1` request) split is the probe's job, not the connector's; a real server needs the
  edit's token span from the delta engine directly.
- The Triton kernel is bf16-only in practice, single-GPU, and untested under TP.

**Scope of the measurements**

- **One model family (Qwen3).** 14B-FP8 for serving, 8B/0.6B bf16 in HF, 3.5-4B for the hybrid. The
  question tail is built with the model's own chat template, so numbers are not portable to a different
  template.
- The length sweep is one session shape (uniform ~597-token filler messages), one edit
  (`--edit-count 1`), six points per line and **one run per cell** — enough to see a slope, not to bound
  curvature. The 24k/30k copy-bandwidth drop could be run-to-run variance. The k-dependence from the
  multi-span entry stacks on top of it and was not re-measured there.
- The Triton re-measure covers **one point of the sweep (30k)**, so the new slope in context length is
  not measured, only its largest point.
- The k-sweep's **k ≈ 17 break-even is an extrapolation from four points**, not a measured crossing.
- The distribution eval is **two seeds, one model**. Sessions are synthetic — real repo text, templated
  turn structure. Cell sizes are 38–40 items; the two items over KL 0.2 are 2 events. `continue-code` is
  n=10–14 and `governing × obey` is n=12. Only the three planted codes are hard-graded; `summarise`,
  `obey`, `continue-code` and `model-written` are graded solely by agreement with full recompute.
- The 2×2 holds δ near 0 by design, so it says nothing about a **large governing edit**; `rewrite` is
  the only large-δ kind and it is non-governing. `mid-governing` is one synthetic construction of a
  governing span that is not the system prompt.
- **Exact-match has a ~20% reference-instability floor** (`prefix-equiv` 0.80–0.86 overall, 0.67 on
  `obey`), which is why KL carries every argument.
- Serving-path parity is **one planted fact plus greedy-text equality** against prefix mode. vLLM's
  offline API cannot give teacher-forced KL at 14B, so the HF work carries quality and the vLLM work
  carries cost.
- **Chunking noise on greedy tokens is real and is not a reuse artefact.** Three phase structures over
  byte-identical token ids give three different first tokens; free-running divergence at 40+ tokens is
  generic and no better at 18% recompute.
- **Byte identity is not context identity.** The matcher can match a block against an identical passage
  elsewhere. `reuse_plan._match` now takes the nearest candidate; `token_segments` in the HF prototype
  remains exposed, and the cheap mitigations (require the predecessor to match, or prefer smallest |δ|)
  are not implemented there.
- **The relocation rule is binary and conservative.** It refuses all relocations on the evidence of one
  |δ| ≈ 10k failure; nothing measures where between δ = 186 (safe) and δ = 10,153 (broken) the boundary
  sits.
- The governing flag is likewise conservative: 71/78 governing items would have been fine reused.
- Hybrid: **three scenarios, one session shape, single runs, no distribution eval.** HF eager prototype,
  not a serving engine. Facts graded on a 12-token forced continuation. M in first-M is uncalibrated.
  δ=+701 because the builder's `grow` argument is characters, not tokens. The old-turn capture prefills
  in two chunks, so its state differs from a one-shot prefill in the last floating-point bits. Hybrid
  memory is **18×** dense.
- `combined` at 0.6B broke under plain `reuse-all` (answered `5111-SIGMA` for `5111-DELTA`, fixed by
  first-32); it did not break at 8B, so it may be a small-model artefact rather than a reproduced
  failure.

**Environment**

- WSL2 has no UVA, so `VLLM_USE_V2_MODEL_RUNNER=0` is required; no nvcc, so a prebuilt
  `flashinfer-jit-cache` wheel is used and LMCache is built against nvcc bundled inside torch's cu13
  wheels.
- The box is shared. Several runs were contended by another agent's GPU work (5–40 s spikes) and were
  discarded and re-run; contended rows are marked in the notebook and excluded here. The length sweep
  was run against a **pinned** copy of the package (`scripts/phase1_probe_pinned.sh` + `MARATHON_SNAP`)
  after a concurrent edit killed a sweep mid-run.
- One harness bug invalidated a first run and is worth remembering: the probe rendered history as a
  plain `role: content` transcript, so the model *continued the log* instead of obeying it, silently
  turning every instruction-following test into a no-op. All quoted numbers are post-fix, through the
  model's own chat template.

**Interpretive corrections already made in the notebook**

- "First output divergence measured" (grow-edit token flip) was retracted — it is chunking noise.
- "Governing is a proxy for at-the-front" was retracted by the 2×2.
- The mechanism "the edited instruction steers generation" still does not hold: for two runs in a row,
  governing damage lands on the *fact* questions, not the instruction-following one. What survives is
  the flag's predictive value, not its stated cause.
- The sweep's "copy is 41% of the edit turn" is superseded by the slab store and the Triton kernel.

---

## 6. Related work

[related.md](related.md) surveys the field: CacheBlend and its descendants (EPIC, Cache-Craft, MPIC),
PromptCache and KVLink's precomputed-module concatenation, Block-Attention's fine-tuned block decoupling,
LMCache and vLLM APC / SGLang RadixAttention as production baselines, the 2025–26 RoPE re-rotation
line (Kamera, KV Packet, SemPIC, Jet-Long), and the closest-in-domain agent work (CacheWise, TokenDance,
Leyline). What is new here is the **composition**, not any single mechanism. Exact RoPE re-rotation is
prior art; a byte-level delta engine is prior art; neither has been joined to the other. Specifically:
(1) reusable spans are identified by an exact byte diff over a **canonical, content-addressed ledger**,
so the reuse decision is ground truth rather than a chunking heuristic or an embedding similarity, and
the same bytes that drive reuse are verified against a full-replay correctness gate; (2) the
"unchanged but repositioned" case is handled by an **exact** rotation rather than partial recompute —
the recompute fraction is not a tuned knob but a consequence of the diff; (3) the plan carries a
*policy*, distinguishing shifted spans (reuse) from relocated ones (recompute) and flagging governing
spans for repair, a distinction the survey's systems do not draw; and (4) it is measured on a live
serving stack against both prefix caching and CacheBlend's own reference implementation. The negative
result on LMCache 0.5.3 in Section 4e is, as far as this survey found, not on record elsewhere.

---

## 7. Reproduction

Environment: WSL2 Ubuntu 24.04 on an RTX 5090, `~/marathon-venv` with torch 2.13.0+cu130, vLLM 0.27.1,
LMCache 0.5.3 (from source), transformers 5.15.0, Triton 3.7.1, `VLLM_USE_V2_MODEL_RUNNER=0`.
`scripts/phase1_setup.sh` builds it.

| result (§) | command |
|---|---|
| API cache hit / collapse (2) | `python -m marathon.live_probe --turns 12`; `--turns 16 --edit-at 13` |
| local prefix-cache collapse (2) | `scripts/phase1_probe.sh --mode prefix --turns 24 --edit-at 20` |
| offline delta bench | `python -m marathon.bench --turns 50 --growth 400 --edit-every 10` |
| north-star sweep (4a) | `scripts/phase1_lengthsweep.sh` (pinned: `scripts/phase1_probe_pinned.sh`) |
| single edit, 14B (4b) | `scripts/phase1_probe.sh --mode {shift,prefix} --turns 24 --edit-at 20 --parity-tokens 8` |
| mid-history / grow (4b) | same, plus `--edit-turn 10` or `--edit-grow 200` |
| repair-first control (4b) | `scripts/phase1_probe.sh --mode shift … --edit-grow 200 --repair-first {0,64,256,1024,16384}` |
| multi-span cost (4c) | `scripts/phase1_multispan.sh` (`--edit-count {1,2,4,8}`, `--move`, `--reuse-moved`) |
| multi-span quality (4c) | `scripts/kvshift_probe.sh --model Qwen/Qwen3-8B --turns 20 --scenario multi-k1,multi-k2,multi-k4,multi-k8,move,combined` |
| HF prototype (4d) | `scripts/kvshift_probe.sh --model Qwen/Qwen3-0.6B --turns 20`; `python -m marathon.kvshift_probe --model Qwen/Qwen3-8B --turns 20` |
| distribution eval (4d) | `scripts/kvshift_eval.sh --model Qwen/Qwen3-8B --sessions 60 --gen-tokens 32 --seed 1234` and `--sessions 84 --seed 1235` |
| CacheBlend (4e) | `scripts/phase1_build_lmcache.sh` then `scripts/phase1_probe.sh --mode blend --turns 24 --edit-at 20 [--recompute-ratio R] [--blend-prefix]` |
| session isolation (4f) | `scripts/phase1_sessions_rerun.sh` → `marathon.local_probe --sessions 2` |
| kernel micro-benchmark (4f) | `scripts/bench_shift_copy.sh` |
| 30k re-measure (4f) | `scripts/bench_shift_30k.sh` |
| hybrid (4g) | `scripts/kvshift_hybrid.sh --turns 20 --max-new-tokens 12 --first-m 256` (`--scenario edit-turn0` adds `replay-mix+first256`) |
| CI gates | `pytest` — includes `tests/test_replay_gate.py` (delta state == full replay at every turn), `tests/test_kvshift.py` (RoPE identity), `tests/test_reuse_plan.py`, `tests/test_shift_store.py`, `tests/test_shift_kernels.py` (GPU-only) |

Knobs that matter: `MARATHON_STORE_TOKENS` (default 32768) and `--store-tokens` / `--gpu-util`;
`MARATHON_NO_TRITON` forces the torch copy fallback; `MARATHON_SNAP` pins `src/marathon` at git HEAD so
a concurrent edit cannot invalidate a run in flight.

---

## 8. What's next

**Phase 2 — cold tier and recall-on-miss.** Demote deep history to embeddings/summaries with a paging
policy and promote it back when a diff or query touches it. Content addressing makes promotion exact:
cold content is a pointer to verifiable bytes, not a paraphrase. Exit criteria: bounded active-window
size on unbounded sessions, recall-on-miss correctly restoring demoted content on targeted questions,
quality delta versus full replay within tolerance.

**Nearer-term systems work, in rough order of value.** Make the copy disappear rather than merely fast —
write re-rotated K in place, or have the attention kernel apply δ at read time. Take the reuse plan into
the connector directly so `k` segments are one request rather than `k+1`, which removes the ~66 ms/edit
term and the k ≈ 17 break-even with it. Then the untested interactions that gate production: tensor
parallelism, preemption, chunked prefill interleaved with a load, and eviction under real pressure. A
CPU/disk tier for the store. Calibrate where between δ = 186 and δ = 10,153 relocation actually becomes
unsafe, so the binary rule can be relaxed. On the hybrid side: a fused scan kernel, a distribution eval,
and a calibrated M.

**Phase 3 — the trust contract (the research bet).** Everything measured here is a systems result: the
model still receives a full prompt and still attends to all of it; only the *computation* is skipped.
The contract DESIGN.md proposes is stronger — the model is told that anything absent from the diff is
unchanged by definition, and stops spending attention re-deriving it. That needs fine-tuning on
delta-formatted interactions with consistency rewards, an eval suite that checks accuracy against
full-context replay while consuming only diff + input tokens, and red-teaming for baseline poisoning,
since a model that treats the substrate as settled truth will treat poisoned substrate the same way.
The regression rule does not change at any phase: **efficiency that changes answers is a regression, not
a win.**

@acrosley 2026-08-19
