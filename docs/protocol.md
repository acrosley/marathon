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

## Server obligations

On receipt, the server resolves `baseline_hash` in its content-addressed store — an unknown hash is `UnknownBaselineError`, and the correct recovery is to request a full snapshot, never to guess. It applies the delta and verifies `sha256(result) == target_hash`; a mismatch is `IntegrityError` and the payload must be rejected — reconstruction is proven, never assumed. Only a verified target is promoted into the store as the next baseline.

## Invariants

Idempotence: replaying a payload against the same baseline yields the same target and the same store state. Determinism: two servers with the same store and payload reach byte-identical states. Concurrency: v0 assumes a single writer per session; the ledger's chain hashes make forks detectable but v0 does not resolve them.

## Future versions

v1 candidates, in rough priority order: signed payloads (snapshot signatures over `target_hash`), compression of insert data (zstd) negotiated per session, chunked baselines for very large states (Merkle segmentation so partial invalidation doesn't rehash everything), and an explicit full-snapshot op for baseline recovery.

@acrosley 2026-08-17
