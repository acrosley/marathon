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

@acrosley 2026-08-18
