# Marathon

**Delta-encoded context architecture for LLMs.** Per-turn cost proportional to *what changed*, not to total context size — sessions that run long because they never re-run the whole race each turn.

Today's LLM interaction model retransmits and re-processes the entire conversation history every turn, even though almost all of it is byte-identical to the previous turn. Marathon replaces that with a deterministic state ledger: a trusted, content-addressed baseline of everything unchanged, carried as reusable state, with only a byte-matched diff plus the new input crossing the wire. In active-inference terms, the model ingests only prediction error — compute scales with surprise.

The founding concept is in [DESIGN.md](DESIGN.md) (doc 0001). The phased roadmap and current status live in [docs/PLAN.md](docs/PLAN.md); measured results are logged in [docs/findings.md](docs/findings.md). The wire format is specified in [docs/protocol.md](docs/protocol.md).

## Status

Phase 0 (systems groundwork) is done; Phase 1 (self-hosted vLLM, warm-tier KV reuse) has met its exit criterion. This repo currently provides the deterministic core: canonical byte-stable serialization with an append-only history guarantee, a hash-chained snapshot ledger with tamper detection, an rsync-style byte-matched delta engine, a turn protocol with cryptographic integrity verification, and a benchmark harness that measures delta savings against full resend. A session runner drives real conversations through the ledger, and a CI gate proves delta-reconstructed state is byte-identical to full-context replay at every turn. Phase 1 is written up end to end — method, results, limitations, reproduction — in [docs/phase1-report.md](docs/phase1-report.md).

**Measured so far** (details in [docs/findings.md](docs/findings.md)). Phase 0, offline: a 50-turn session with edits every 10 turns resends 88.85% fewer bytes than full resend. Live against the Anthropic API (Haiku 4.5), append-only canonical history hits the prompt cache exactly (3 uncached tokens per turn), and a one-word edit to the first message drops cache reads to zero — the gap Phase 1 targeted.

Phase 1 closes it. On self-hosted vLLM 0.27 (Qwen3-14B-FP8, one RTX 5090), **position-shifted KV reuse** — the delta engine locates the edit, unchanged history is reused with cached keys re-rotated by the token offset (an exact RoPE identity), and only the edited span plus the new input are prefilled — makes the edit turn's prefill nearly flat in context length while prefix caching's grows linearly:

| history at edit turn | prefix caching | Marathon shift | speedup |
|---:|---:|---:|---:|
| 4k tokens | 0.42 s | 0.15 s | 2.7× |
| 12k | 1.27 s | 0.16–0.20 s | 6.5–7.6× |
| 16k | 1.78 s | 0.20 s | 8.9× |
| 30k | 3.94 s | 0.26 s | **15.1×** |

Unchanged turns keep the prefix-cache fast path (identical between modes); every turn's output is byte-identical to prefix mode; several edited messages per turn cost +66 ms each rather than a context-scaled recompute; moved blocks are recomputed rather than transplanted. Across 144 realistic sessions on Qwen3-8B, re-rotated reuse forwards 1.5% of tokens for median KL 0.003 against full recompute (prefix caching forwards 66% for KL 0.0015); the one failure class is edits to *governing* spans (system prompt / standing instructions), which the delta layer's `reuse_plan` flags. CacheBlend (LMCache) on the same stack only tied full recompute. The KV connector is session-keyed with LRU eviction and a fused Triton copy kernel at 1.43 TB/s.

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

### End-to-end server (needs a GPU and vLLM)

`marathon.server` is the whole pipeline: it verifies a turn payload against its
content-addressed store, plans KV reuse against the session's previous state, and drives
vLLM with the shift connector. `marathon.client` keeps the history and ships only deltas.

```bash
python -m marathon.server --model Qwen/Qwen3-0.6B --port 8000   # POST /v1/turn
python scripts/server_demo.py --url http://127.0.0.1:8000       # 12 turns + a mid-history edit
bash scripts/server_demo.sh --model Qwen/Qwen3-14B-FP8 --gpu-util 0.80  # both, end to end
```

```python
from marathon.client import Client, http
c = Client(http("http://127.0.0.1:8000"))
c.turn("s1", "Remember the access code 7391-KAPPA.")
c.edit("s1", 0, "Remember the access code 5820-OMEGA.")  # takes effect next turn
print(c.turn("s1", "What is the access code?"))          # reply + per-turn metrics
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
  local_probe.py  Phase 1 probe: self-hosted vLLM, modes none / prefix / blend (LMCache) / shift (Marathon)
  kvshift.py      position-shifted KV reuse: RoPE re-rotation of cached keys, delta-located spans, stitching (HF prototype)
  vllm_shift_connector.py  vLLM KV connector implementing shifted reuse inside the paged KV cache
  shift_store.py  session-keyed position-indexed KV store with LRU eviction
  shift_kernels.py  fused Triton scatter+re-rotate copy kernel
  reuse_plan.py   delta layer -> P/E'/S segments + policy (reuse / repair for governing edits / full)
  server.py       end-to-end server: verify payload -> plan reuse -> drive vLLM -> reply + metrics
  client.py       the other half: keeps the history, ships deltas, in-process or over HTTP
  kvshift_eval.py distribution-level quality eval (sessions x edit kinds x conditions)
  stitch_train.py Phase 3: LoRA consistency fine-tuning against stitched caches (self-distillation to the full-recompute teacher)
scripts/          Phase 1 WSL2 environment: setup, LMCache source build + patches, probe runner
tests/            full test suite incl. randomized diff round-trip properties
docs/             plan and protocol spec
DESIGN.md         founding design doc (0001)
```

## Design invariants

Determinism first: identical logical state must always serialize to identical bytes, so state can be referenced by hash instead of resent. Never trust reconstruction: every resolved turn is verified against the declared target hash before it becomes the next baseline. Append-only history: serializing `history[:k]` is always a byte-prefix of serializing `history[:k+1]`, which is what makes provider prefix caching hit maximally in Phase 0. Efficiency must not change answers: correctness is always evaluated against full-context replay, which the deterministic ledger makes exact.

@acrosley 2026-08-17
