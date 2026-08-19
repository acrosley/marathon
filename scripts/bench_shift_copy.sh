#!/usr/bin/env bash
# Micro-benchmark the connector's re-rotate-and-scatter: torch path vs fused Triton.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/bench_shift_copy.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/marathon-logs"
log="$HOME/marathon-logs/bench_shift_copy.log"
cd "$here/.."
PYTHONPATH="$here/../src" "$HOME/marathon-venv/bin/python" "$here/bench_shift_copy.py" "$@" 2>&1 | tee "$log"
echo "log=$log"
