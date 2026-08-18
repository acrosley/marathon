# Marathon

**Delta-encoded context architecture for LLMs.** Per-turn cost proportional to *what changed*, not to total context size — sessions that run long because they never re-run the whole race each turn.

Today's LLM interaction model retransmits and re-processes the entire conversation history every turn, even though almost all of it is byte-identical to the previous turn. Marathon replaces that with a deterministic state ledger: a trusted, content-addressed baseline of everything unchanged, carried as reusable state, with only a byte-matched diff plus the new input crossing the wire. In active-inference terms, the model ingests only prediction error — compute scales with surprise.

The founding concept is in [DESIGN.md](DESIGN.md) (doc 0001). The phased roadmap and current status live in [docs/PLAN.md](docs/PLAN.md); measured results are logged in [docs/findings.md](docs/findings.md). The wire format is specified in [docs/protocol.md](docs/protocol.md).

## Status

Phase 0 (systems groundwork, no custom model required) is under way. This repo currently provides the deterministic core: canonical byte-stable serialization with an append-only history guarantee, a hash-chained snapshot ledger with tamper detection, an rsync-style byte-matched delta engine, a turn protocol with cryptographic integrity verification, and a benchmark harness that measures delta savings against full resend. A session runner drives real conversations through the ledger, and a CI gate proves delta-reconstructed state is byte-identical to full-context replay at every turn.

**Measured so far** (details in [docs/findings.md](docs/findings.md)): offline, a 50-turn session with edits every 10 turns resends 88.85% fewer bytes than full resend, with wire cost flat at ~1.25 KB/turn while state grows to 22 KB. Live against the Anthropic API (Haiku 4.5), append-only canonical history hits the prompt cache exactly — whole history read from cache, only the new turn written, 3 uncached tokens per turn. A one-word edit to the first message drops the provider's cache reads to zero and re-processes the full history, while Marathon's own wire payload absorbs the same edit in ~+560 bytes — the gap Phase 1 targets.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
python -m marathon.bench --turns 50 --growth 400 --edit-every 10
```

Example of what the benchmark demonstrates: over a 50-turn session, full resend scales quadratically in total bytes while Marathon's wire cost stays near-flat per turn, including when earlier messages are edited — the case that invalidates naive prefix caching.

To measure real-world prefix-cache behavior against the Anthropic API (optional, costs a small amount):

```bash
pip install -e ".[live]"
ANTHROPIC_API_KEY=... python -m marathon.live_probe --turns 6
```

## Layout

```
src/marathon/
  canonical.py    byte-stable serialization + content addressing
  ledger.py       append-only hash-chained snapshot ledger
  diff.py         rsync-style byte-matched block delta engine
  protocol.py     turn payload {baseline_hash, target_hash, delta, new_input}
  session.py      session runner: canonical bytes are the single path to the wire
  bench.py        offline full-resend vs delta benchmark
  live_probe.py   experimental TTFT / prompt-cache probe (Anthropic API)
tests/            full test suite incl. randomized diff round-trip properties
docs/             plan and protocol spec
DESIGN.md         founding design doc (0001)
```

## Design invariants

Determinism first: identical logical state must always serialize to identical bytes, so state can be referenced by hash instead of resent. Never trust reconstruction: every resolved turn is verified against the declared target hash before it becomes the next baseline. Append-only history: serializing `history[:k]` is always a byte-prefix of serializing `history[:k+1]`, which is what makes provider prefix caching hit maximally in Phase 0. Efficiency must not change answers: correctness is always evaluated against full-context replay, which the deterministic ledger makes exact.

@acrosley 2026-08-17
