#!/usr/bin/env bash
# Depth test for the Phase 2 / Phase 1 composition: a paged session is a front-of-view
# edit on every single turn. Runs the same demote-driven conversation twice -- shift
# connector on, then off -- and compares the generated text turn by turn. Any turn where
# they differ is a turn where reuse changed the answer.
#
# Usage: wsl -d Ubuntu -- bash /mnt/c/.../scripts/paged_depth.sh [--model M] [--turns N]
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
model="Qwen/Qwen3-0.6B"
gpu_util="0.30"
mml="16384"
turns="30"
demote="4"
store="16384"
maxtok="24"
maxstale=""
repair=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2;;
    --gpu-util) gpu_util="$2"; shift 2;;
    --max-model-len) mml="$2"; shift 2;;
    --turns) turns="$2"; shift 2;;
    --demote) demote="$2"; shift 2;;
    --store-tokens) store="$2"; shift 2;;
    --max-tokens) maxtok="$2"; shift 2;;
    --max-stale) maxstale="$2"; shift 2;;
    --repair-first) repair="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
out="$HOME/marathon-logs"
mkdir -p "$out"
common=(--model "$model" --gpu-util "$gpu_util" --max-model-len "$mml" --turns "$turns"
        --demote "$demote" --store-tokens "$store" --max-tokens "$maxtok" --fixed-replies --fact-probe
        ${maxstale:+--max-stale "$maxstale"} ${repair:+--repair-first "$repair"})

bash "$here/server_demo.sh" "${common[@]}" --json "$out/paged_reuse.json"
bash "$here/server_demo.sh" "${common[@]}" --no-reuse --json "$out/paged_control.json"

"$HOME/marathon-venv/bin/python" - "$out/paged_reuse.json" "$out/paged_control.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["rows"]
b = json.load(open(sys.argv[2]))["rows"]
print(f"\n{'turn':>5} {'prompt':>7} {'reuse_s':>8} {'ctrl_s':>8} {'reused':>7} {'ph':>3}  text match")
bad = []
for x, y in zip(a, b):
    same = x["reply"] == y["reply"]
    if not same:
        bad.append(x["turn"] if "turn" in x else a.index(x))
    print(f"{a.index(x):>5} {x['prompt_tokens']:>7} {x['prefill_s']:>8} {y['prefill_s']:>8} "
          f"{x['reused_tokens']:>7} {x['phases']:>3}  {'ok' if same else 'DIVERGED'}")
ps = sorted(r["prefill_s"] for r in a)
qs = sorted(r["prefill_s"] for r in b)
print(f"\nprefill p50 reuse={ps[len(ps)//2]:.3f}s max={ps[-1]:.3f}s | "
      f"control p50={qs[len(qs)//2]:.3f}s max={qs[-1]:.3f}s")
for name, rows in (("reuse", a), ("control", b)):
    sc = [r for r in rows if r.get("hit") is not None]
    if sc:
        h = sum(1 for r in sc if r["hit"])
        print(f"{name:>8} fact exact-match: {h}/{len(sc)} = {h / len(sc):.3f}")
print(f"turns whose text diverged: {len(bad)}/{len(a)}" + (f" first at {bad[0]}" if bad else ""))
sys.exit(1 if bad else 0)
PYEOF
