#!/usr/bin/env bash
# Phase 2 x Phase 1 composition test: the paged 14B workload with the churn ceiling at
# two values, against a connector-off control. Scored turns are sampled on two
# consecutive turns so they cannot phase-lock onto the reuse/refresh alternation, and
# the store is pre-sized from the active window so no save has to grow a buffer.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$HOME/marathon-logs"
mkdir -p "$out"

model="${MODEL:-Qwen/Qwen3-14B-FP8}"
common=(--model "$model" --gpu-util 0.93 --max-model-len 40960 --store-tokens 24576
        --active-window 8192 --turns "${TURNS:-40}" --demote 0 --max-tokens 16
        --fixed-replies --fact-probe)

run() {  # name, extra args...
  local name="$1"; shift
  rm -f "$out/churn_$name.json"
  echo "=== condition: $name ($*) ==="
  bash "$here/server_demo.sh" "${common[@]}" "$@" --json "$out/churn_$name.json"
  sleep 20   # let the engine hand the card back before the next one claims it
}

#run control --no-reuse
run mc025 --max-churn 0.25
run mc100 --max-churn 1.0
echo "done: $out/churn_{control,mc025,mc100}.json"
