# Marathon Turn Protocol — v0

Wire specification for the delta-encoded turn payload. Implemented in `src/marathon/protocol.py` and `src/marathon/diff.py`.

## Content addressing

All state is addressed as `sha256:<64 hex chars>` over its canonical bytes. Canonical bytes are produced by `canonical.canonical_bytes`: JSON with sorted keys, separators `(",", ":")`, `ensure_ascii=False`, UTF-8 encoded, NaN/Infinity forbidden. Histories serialize via `canonical.serialize_history` as canonical JSON Lines, which guarantees the append-only property: `serialize_history(h[:k])` is a byte-prefix of `serialize_history(h[:k+1])`.

## Turn payload

```json
{
  "v": 0,
  "baseline_hash": "sha256:…" | null,
  "target_hash": "sha256:…",
  "delta": { "v": 0, "block_size": 64, "ops": [["c", offset, length], ["i", "<base64>"]] },
  "new_input": "…"
}
```

`baseline_hash` is `null` on the first turn of a session, meaning the empty baseline (zero bytes). `target_hash` is the content address of the state the client is asserting after this turn. `delta` transforms the baseline bytes into the target bytes. `new_input` is the fresh user input for this turn.

## Delta ops

`["c", offset, length]` copies `length` bytes from the baseline starting at `offset`. `["i", "<base64>"]` inserts literal bytes. Applying the ops in order concatenates their outputs to produce the target. Copies out of the baseline's range are a hard error. `block_size` records the granularity used by the matcher; it is informational for the applier (application is fully determined by the ops) but required for reproducing the delta.

## Reuse plan

The delta carries enough information to tell the KV layer what to do, not just what changed. `reuse_plan.plan(old_state, new_state, tokenize, head_tokens=…)` classifies a transition and emits a **list of reusable segments**. Position-shifted KV reuse — keep whatever runs of history survived, recompute only the rewritten entries, and re-rotate each surviving run's keys by *its own* shift — is exact for position but approximate for attention: a reused run's KV attended to the *old* text before it. Measurement (findings, 2026-08-18) puts the boundary at the kind of span edited, not at whether later text depends on it: fact-carrying edits survive plain reuse, because the query attends to the rewritten span directly and the reused text only carries the pointer, while edits inside a **governing** span — system prompt, standing instructions, persona, output-format or language directives, tool policy — moved the first generated token by 30x the KL of every other case. A message is governing when it serializes `"governing": true`; the `system` role gets it by default, and the key is written only when true, so canonical bytes of sessions that never set it are unchanged.

Matching is done on canonical JSONL lines — the ledger's own unit — so a segment is always a whole run of entries and never needs snapping to a token boundary. Where several old entries are byte-identical (every bare `"ok"` acknowledgement is the same line), the *nearest* one is matched, not the first: byte identity is not context identity, and a reused entry's KV encodes whatever preceded it. It is order-agnostic: an entry that *moved* is still the same entry, so a segment's source range may run backwards relative to its neighbours', and its `delta` may be negative. `tokenize` maps one line to the token ids it contributes to the prompt, and `head_tokens` is the length of anything the prompt puts before the first line, so the coordinates are the serving layer's.

- `segments` — the reused runs in destination order. Each is a `Segment(src_start, src_end, dst_start)` with `delta = dst_start - src_start` and `length`. Everything they do not cover is recomputed.
- `moved` — parallel to `segments`: whether that run *relocated* (its entries' index in the history changed) rather than merely shifting because something before it changed length. Measurement (findings, 2026-08-19) is that a shift is safe to re-rotate and a relocation is not — a block that moved 10k positions has KV summarising a completely different prefix, which re-rotation cannot repair and a leading-chunk recompute does not fix — so relocated runs are recomputed by default.
- `total` — length of the new sequence in the same token coordinates.
- `policy` — `reuse` (stitch and go), `repair` (stitch, but natively recompute the first `repair_first` tokens of every non-leading segment so they attend to the new text before them), `full` (nothing survived; recompute).
- `repair_first` — leading tokens of each non-leading segment to recompute; 0 unless `policy` is `repair`, default 256.
- `reason` — why this policy was chosen.
- `p`, `e_start`/`e_end`, `delta`, `s_start`/`s_end` — the earlier single-span vocabulary, kept as derived properties: the leading prefix, the first recomputed span, and the last reused segment.

`plan(...).to_kv_transfer_params()` emits a **list** of `{"dst_start", "dst_end", "delta"}`, one per reused segment past the leading prefix, in destination order — `dst_start` already past the repaired head. Relocated segments are omitted; `to_kv_transfer_params(reuse_moved=True)` includes them, which is measured-unsafe and exists only to re-measure it. The leading prefix is omitted because the serving layer's own prefix cache already holds it; every later segment needs the connector even at `delta == 0`, since a recomputed span before it has already broken the prefix. An empty list means there is nothing for the connector to do. Block alignment stays with the caller, since vLLM only counts matched tokens in whole blocks — and because that API can only express matched tokens as a *prefix*, `k` segments are handed over as `k + 1` requests (`local_probe._phases`), each stopping on the block boundary where the next segment begins.

## Connector

`marathon.vllm_shift_connector.MarathonShiftConnector` is the serving-side consumer of a reuse plan inside vLLM 0.27. Every request carries `{"session": "<id>", "load": {...}, "save": true}` in `kv_transfer_params`; `marathon.shift_store` holds the bookkeeping, which is what the CPU tests cover.

What is safe now. The KV store is **keyed by session id**, one position-indexed buffer per session per layer, so two sessions can never read each other's KV; a request with no session id is pass-through, behaving exactly as if no connector were configured. **One writer per session** is enforced, not assumed: if a second in-flight request claims a session the table refuses it, the request is served with no reuse at all, and a load therefore can never overlap an in-flight save on the same session. **Saves are per scheduler step** — only the positions the request actually computed (`num_computed_tokens` onward, gathered after each layer is written), so chunked and continued prefills are recorded correctly, and a save at a lower `dst_start` truncates the positions above it, which is what an edit means. **Memory is bounded**: a total budget in tokens (`MARATHON_STORE_TOKENS`, default 32768; 164 KB/token on Qwen3-14B) with LRU eviction of whole sessions. Because eviction happens on the worker, the scheduler side mirrors the same bookkeeping without tensors and **declines** a load whose source positions were evicted or truncated — a miss costs a recompute, never a wrong answer. `stats()` reports tokens per session, hits, misses, loads, saves and evictions.

What is still not safe, honestly. Tensor parallelism is untested (each worker would keep its own store, and nothing coordinates them). Preemption is untested: a request preempted and recomputed mid-flight keeps its store positions, and nothing has measured whether the scheduler's re-issue lands on the same coordinates. Chunked prefill *interleaved with* a load on the same request is untested — the probe always hands a load to a request whose local prefix hit already ends at `dst_start`. Reuse is still whole-block only, so a ragged head or tail of a reused run is recomputed. And the store is GPU-resident with no CPU/disk tier, so the budget is a hard ceiling rather than a paging policy.

## Server obligations

On receipt, the server resolves `baseline_hash` in its content-addressed store — an unknown hash is `UnknownBaselineError`, and the correct recovery is to request a full snapshot, never to guess. It applies the delta and verifies `sha256(result) == target_hash`; a mismatch is `IntegrityError` and the payload must be rejected — reconstruction is proven, never assumed. Only a verified target is promoted into the store as the next baseline.

## Invariants

Idempotence: replaying a payload against the same baseline yields the same target and the same store state. Determinism: two servers with the same store and payload reach byte-identical states. Concurrency: v0 assumes a single writer per session; the ledger's chain hashes make forks detectable but v0 does not resolve them.

## Future versions

v1 candidates, in rough priority order: signed payloads (snapshot signatures over `target_hash`), compression of insert data (zstd) negotiated per session, chunked baselines for very large states (Merkle segmentation so partial invalidation doesn't rehash everything), and an explicit full-snapshot op for baseline recovery.

@acrosley 2026-08-17
