# Single-request multi-segment KV load, and generation-0 reuse

**Status:** designed and implemented, **not yet run on hardware** (written 2026-08-21, no GPU
until 2026-08-24). Two independent pieces, aimed at the two surviving explanations for the
paged workload's answer-level collapse (findings 2026-08-21).

## The problem

vLLM's KV-connector API expresses externally supplied KV only as a **prefix**.
`get_num_new_matched_tokens` returns a count; the scheduler folds it into a scalar
`num_computed_tokens`; the model runner prefills the contiguous suffix
`[num_computed, num_computed + num_scheduled)`. A reuse plan is not a prefix — it is `k`
surviving runs with `k` fresh gaps between them — so `reuse_plan.phases` hands the segments
over as `k + 1` sequential requests, each loading one segment and leaning on the engine's
prefix cache to carry the earlier ones.

That path is where the answers go. Measured 2026-08-21: the stitched KV itself is correct to
bf16 in every layer (`MARATHON_VERIFY_LOAD`, max relative error 2–3e−3 across all 28 layers of
Qwen3-0.6B), yet fact exact-match falls from 0.75 to 0.35 on Qwen3-8B, and capping the plan at
one segment recovers a third of the gap. Phase 3's HF experiment stitches every segment into
one cache, runs a single forward, and loses nothing (250/250).

## Design 1 — non-prefix matches, one request

**Why it is sound.** Attention reads earlier positions out of the paged cache through the block
table. A token therefore has to be *present*, not *computed in the same forward pass* as the
tokens attending to it. The connector writes reused spans in `start_load_kv`, which runs before
the forward; gaps earlier in the same batch are written to their slots before later gaps attend
to them. So prefilling only the gaps is legitimate — the API contract is the only obstacle, not
the mechanism.

**Where the contract actually lives.** One line in `v1/worker/gpu_model_runner.py`:

```python
positions_np = self.input_batch.num_computed_tokens_cpu[req_indices] + self.query_pos.np[...]
```

That is the sole place the contiguity assumption enters input construction. Token ids, slot
mapping, block tables and attention metadata are all derived from `positions_np`, so making
that array explicit is the whole change.

**The patch** (`scripts/patch_vllm_gapfill.py`, idempotent, `--revert`, anchors asserted exactly
once so a vLLM upgrade fails loudly):

1. `v1/core/sched/scheduler.py` — after the external-match count is folded in, take any gap plan
   the connector offered and republish it for the runner. `num_computed_tokens` becomes
   *matched + gaps already computed*, which keeps every existing "done prefilling" comparison in
   that file correct, including under chunked prefill.
2. `v1/worker/gpu_model_runner.py` — for a request carrying a gap plan, overwrite its slice of
   `positions_np` with the explicit positions.

`marathon.gapfill_channel` is the hand-off: `offer` / `take` / `publish` / `active` / `release`.
It is a module-level dict, which is legitimate only because the v1 scheduler and worker share a
process on a single GPU. **Tensor parallelism is not supported** — each worker would hold its own
copy.

**The arithmetic** is `marathon.gapfill`, pure and unit-tested (23 tests): block-align the plan's
loads, drop anything that cannot survive alignment (rather than rounding outward onto positions
the connector will not fill), merge, fold in the engine's own prefix hit, and emit the exact
complement as the positions to compute. `check()` asserts the invariants the patch relies on —
spans sorted and disjoint, the partition exact, and the engine always left the final position,
or there is no forward pass to run and nothing to sample. A test pins that `align()` clips
identically to `phases()`, so the two paths reuse the *same* spans and an A/B measures the
request structure and nothing else.

**Wiring.** `MARATHON_GAPFILL=1` plus `--gapfill`: the server sends every segment as
`kv_transfer_params["loads"]` in one request and issues no warm-up phases; the connector
coverage-checks each segment, drops what the store cannot serve, and offers the gap plan. Without
the patch the scheduler ignores the channel, so the connector declines rather than guessing.

**Untested on hardware.** The risks are the scheduler's budget accounting when the gap count
exceeds one step, and any place other than `positions_np` that assumes contiguity (attention
metadata construction is the one to watch).

## Design 2 — generation-0 reuse, and why it is blocked

The compounding hypothesis: an edit turn saves with `"full"`, re-gathering the whole prompt out
of the paged cache *including the span the connector just stitched there*. After two reuse turns
the store holds a rotated copy of a rotated copy, indistinguishable from freshly computed KV, so
the next turn reuses it believing it is fresh.

`marathon.remap` is the alternative: never re-save a reused span, and keep an address book from
logical position to store index. The invariant that makes it work is the store's existing one —
**store index `i` holds keys as computed at position `i`** — so a span now at logical `p` needs
one rotation of exactly `p − i`. Offsets compose by addition because rotations do, so a span that
has moved `d1` then `d2` is reachable by a single exact rotation of `d1 + d2`, however many turns
have passed. Ten unit tests cover it, including a twelve-turn paged simulation that checks every
emitted load reaches the span's *generation-0* index (drift 2400 tokens).

**It cannot be wired into the current store, and the reason is structural.** `ShiftStore` is a
flat `[base, filled)` window: a save at a lower `dst_start` truncates everything above it. Under
no-resave the fresh gaps between reused segments are saved at their own — lower — indices every
turn, so each save destroys the generation-0 bytes the remap addresses higher up. Measured on
CPU: 19 truncating writes and 80 corrupted positions over 20 paged turns, caught by the
fingerprint harness. `MarathonServer(resave=False)` therefore raises `NotImplementedError`
rather than offering a silent wrong-answer path, and two tests pin the blocker.

**What unblocking costs.** Per-position validity: an interval-allocating store (a free list plus
an interval map, `covers` answering per range rather than against one high-water mark) and a
scheduler-side mirror that agrees with it. That is a real rewrite of `shift_store`, and it should
not be attempted until Design 1 has been measured — if the single-request path restores exact-match
on its own, compounding was never the binding constraint and this stays unbuilt.

## Order of work

Design 1 first: it is testable in one GPU run against the same workload, it needs no store
changes, and it addresses the mechanism the measurement actually implicates. Design 2 only if
answers are still lost with `k` segments loaded in one request.
