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

The delta carries enough information to tell the KV layer what to do, not just what changed. `reuse_plan.plan(old_state, new_state, tokenize)` classifies a transition into one of three policies. Position-shifted KV reuse — keep the prefix `P`, recompute the edited span `E'`, copy the unchanged suffix `S` with its keys re-rotated by `delta` — is exact for position but approximate for attention: `S`'s KV attended to the *old* span. Measurement (findings, 2026-08-18) puts the boundary at the kind of span edited, not at whether later text depends on it: fact-carrying edits survive plain reuse, because the query attends to `E'` directly and `S` only carries the pointer, while edits inside a **governing** span — system prompt, standing instructions, persona, output-format or language directives, tool policy — moved the first generated token by 30x the KL of every other case. A message is governing when it serializes `"governing": true`; the `system` role gets it by default, and the key is written only when true, so canonical bytes of sessions that never set it are unchanged.

- `p` — unchanged prefix length, in tokens.
- `e_start`, `e_end` — the recomputed span `E'` in new-token coordinates (for an append-only turn, the appended tail).
- `delta` — position shift applied to `S`, i.e. `|E'| - |E|`.
- `s_start`, `s_end` — the reusable span in new-token coordinates; excludes the appended turn.
- `policy` — `reuse` (stitch and go), `repair` (stitch, but natively recompute the first `repair_first` tokens of `S` so they attend to `E'`), `full` (no reuse: multiple disjoint edits, or a truncated history).
- `repair_first` — leading `S` tokens to recompute; 0 unless `policy` is `repair`, default 256.
- `reason` — why this policy was chosen.

`plan(...).to_kv_transfer_params()` emits `{"dst_start", "dst_end", "delta"}`, the payload the vLLM shift connector takes, with `dst_start` already past the repaired head; `None` means do not reuse. Block alignment stays with the caller, since vLLM only counts matched tokens in whole blocks.

## Server obligations

On receipt, the server resolves `baseline_hash` in its content-addressed store — an unknown hash is `UnknownBaselineError`, and the correct recovery is to request a full snapshot, never to guess. It applies the delta and verifies `sha256(result) == target_hash`; a mismatch is `IntegrityError` and the payload must be rejected — reconstruction is proven, never assumed. Only a verified target is promoted into the store as the next baseline.

## Invariants

Idempotence: replaying a payload against the same baseline yields the same target and the same store state. Determinism: two servers with the same store and payload reach byte-identical states. Concurrency: v0 assumes a single writer per session; the ledger's chain hashes make forks detectable but v0 does not resolve them.

## Future versions

v1 candidates, in rough priority order: signed payloads (snapshot signatures over `target_hash`), compression of insert data (zstd) negotiated per session, chunked baselines for very large states (Merkle segmentation so partial invalidation doesn't rehash everything), and an explicit full-snapshot op for baseline recovery.

@acrosley 2026-08-17
