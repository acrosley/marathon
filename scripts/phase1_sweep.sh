#!/usr/bin/env bash
# Phase 1 sweep driver: run local_probe configs back to back, each only once the
# GPU is free (another agent shares this box), with leftover engines killed.
# Usage (from Windows): wsl -d Ubuntu -- bash .../scripts/phase1_sweep.sh
set -u
here="$(dirname "$0")"

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 6000 ] && break
    echo "waiting for gpu: ${used} MiB used"
    sleep 30
  done
}

run() {
  pkill -9 -f EngineCore >/dev/null 2>&1
  wait_for_gpu
  echo "=== $* ==="
  bash "$here/phase1_probe.sh" "$@"
}

run --mode blend --turns 24 --edit-at 20 --parity-tokens 16 --recompute-ratio 0.15
run --mode blend --turns 24 --edit-at 20 --parity-tokens 16 --recompute-ratio 0.05
run --mode blend --turns 24 --edit-at 20 --parity-tokens 16 --recompute-ratio 0.02
run --mode blend --blend-prefix --turns 24 --edit-at 20 --parity-tokens 16 --recompute-ratio 0.15
run --mode blend --blend-prefix --turns 24 --edit-at 20 --parity-tokens 16 --recompute-ratio 0.05
echo "sweep done"
