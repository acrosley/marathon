#!/usr/bin/env bash
# Re-run the 30k point of the length sweep against a pinned snapshot of the working
# tree, to measure the fused Triton copy in the connector. Modes default to both.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$HOME/marathon-snap"; mkdir -p "$HOME/marathon-snap"
cp -r "$here/../src/marathon" "$HOME/marathon-snap/marathon"
common="--turns 54 --edit-at 50 --parity-tokens 16 --gpu-util 0.78"
for mode in "${@:-prefix shift}"; do
  extra=""
  [ "$mode" = shift ] && extra="--store-tokens 33280"
  echo "=== $(date +%T) $mode ==="
  bash "$here/phase1_probe_pinned.sh" --mode "$mode" $common $extra
done
echo "=== $(date +%T) done ==="
