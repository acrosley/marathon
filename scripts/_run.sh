#!/usr/bin/env bash
set -uo pipefail
mkdir -p "$HOME/marathon-logs"
cd "$HOME"
export VLLM_USE_V2_MODEL_RUNNER=0
MODE=$1; shift
N="run_${MODE}_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')"
L="$HOME/marathon-logs/$N.log"
"$HOME/marathon-venv/bin/python" -m marathon.local_probe --mode "$MODE" \
  --model "${MODEL:-Qwen/Qwen3-14B-FP8}" "$@" --json "$HOME/marathon-logs/$N.json" > "$L" 2>&1
echo "== $MODE $* exit=$?"
grep -E '^mode=' "$L"
grep -A40 -E '^ *turn ' "$L" | grep -vE 'INFO|WARNING|it/s'
grep -oE 'shift: (loaded|declining)[^)]*' "$L" | head -5
