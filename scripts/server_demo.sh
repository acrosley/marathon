#!/usr/bin/env bash
# End-to-end Marathon server demo: boot the HTTP endpoint, run a 12-turn conversation
# with mid-history edits through it, print the per-turn table, assert the planted fact.
#
# Usage:
#   wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon-server/scripts/server_demo.sh
#   ... --model Qwen/Qwen3-14B-FP8 --gpu-util 0.80 --max-model-len 32768
set -uo pipefail

model="Qwen/Qwen3-0.6B"
gpu_util="0.30"
max_model_len="16384"
store_tokens="16384"
max_tokens="16"
turns="12"
edit_at="9"
port="8${RANDOM:0:3}"
extra=""
demote=""
json=""
maxstale=""
maxchurn=""
maxseg=""
verify=""
fixed=""
factprobe=""
repair=""
window=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2;;
    --gpu-util) gpu_util="$2"; shift 2;;
    --max-model-len) max_model_len="$2"; shift 2;;
    --store-tokens) store_tokens="$2"; shift 2;;
    --max-tokens) max_tokens="$2"; shift 2;;
    --turns) turns="$2"; shift 2;;
    --edit-at) edit_at="$2"; shift 2;;
    --port) port="$2"; shift 2;;
    --no-reuse) extra="--no-reuse"; shift;;
    --demote) demote="$2"; shift 2;;
    --json) json="$2"; shift 2;;
    --fixed-replies) fixed="--fixed-replies"; shift;;
    --fact-probe) factprobe="--fact-probe"; shift;;
    --max-stale) maxstale="$2"; shift 2;;
    --max-churn) maxchurn="$2"; shift 2;;
    --max-segments) maxseg="$2"; shift 2;;
    --verify-load) verify=1; shift;;
    --repair-first) repair="$2"; shift 2;;
    --active-window) window="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
# the venv's editable install points at the main repo checkout, so this worktree's
# sources have to come first on the path
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
# ponytail: WSL2 has no UVA (pinned host-mapped memory); vLLM 0.27 v2 runner needs it
export VLLM_USE_V2_MODEL_RUNNER=0
[ -n "$verify" ] && export MARATHON_VERIFY_LOAD=1
py="$HOME/marathon-venv/bin/python"
mkdir -p "$HOME/marathon-logs"
tag="serverdemo_$(echo "$model$extra$demote" | tr -c "a-zA-Z0-9" "_")"
log="$HOME/marathon-logs/$tag.log"

echo "=== $model $extra on port $port (server log: $log) ==="
cd "$HOME"
"$py" -m marathon.server --model "$model" --port "$port" --gpu-util "$gpu_util" \
  --max-model-len "$max_model_len" --store-tokens "$store_tokens" \
  --max-tokens "$max_tokens" $extra ${maxstale:+--max-stale "$maxstale"} ${maxchurn:+--max-churn "$maxchurn"} ${maxseg:+--max-segments "$maxseg"} ${repair:+--repair-first "$repair"} ${window:+--active-window "$window"} > "$log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null' EXIT

for _ in $(seq 1 180); do
  grep -q "marathon.server ready" "$log" && break
  kill -0 $srv 2>/dev/null || { echo "server died; tail of $log:"; tail -30 "$log"; exit 1; }
  sleep 2
done
grep -q "marathon.server ready" "$log" || { echo "server never came up"; tail -30 "$log"; exit 1; }

"$py" "$here/server_demo.py" --url "http://127.0.0.1:$port" --turns "$turns" --edit-at "$edit_at" ${demote:+--demote "$demote"} ${json:+--json "$json"} $fixed $factprobe
rc=$?
echo "demo exit=$rc  server log=$log"
exit $rc
