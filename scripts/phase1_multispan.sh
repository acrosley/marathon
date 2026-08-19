#!/usr/bin/env bash
# Part 2 of the multi-span study: the whole prefix-vs-shift matrix, back to back.
# Each local_probe invocation starts its own engine, so this exists to keep the GPU
# busy without a human between runs.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_multispan.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
run() { echo "=== $* ==="; bash "$here/phase1_probe.sh" "$@"; }
common="--turns 24 --edit-at 20 --parity-tokens 16"

# baselines: the same mutation, without any KV reuse
run --mode prefix $common --edit-count 1
run --mode prefix $common --edit-count 0 --move

for k in 1 2 4 8; do
  run --mode shift $common --edit-count "$k"
done
run --mode shift $common --edit-count 0 --move   # a pure move: no in-place edit at all
run --mode shift $common --edit-count 4 --move
