#!/usr/bin/env bash
# Distribution-level quality eval for position-shifted KV reuse (marathon.kvshift_eval).
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/kvshift_eval.sh \
#          --model Qwen/Qwen3-8B --sessions 60
# Writes ~/marathon-logs/<name>.{log,jsonl,summary.json}.
set -uo pipefail
mkdir -p "$HOME/marathon-logs"
name="kvshifteval_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')"
log="$HOME/marathon-logs/$name.log"
cd "$HOME"
# single-tenant GPU: refuse to start if another agent's job is resident
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
need=${KVSHIFT_NEED_MIB:-24000}
if [ "$free" -lt "$need" ]; then
  echo "GPU has only ${free} MiB free (< ${need}); not starting."
  exit 75
fi
"$HOME/marathon-venv/bin/python" -m marathon.kvshift_eval "$@" \
  --jsonl "$HOME/marathon-logs/$name.jsonl" \
  --summary "$HOME/marathon-logs/$name.summary.json" > "$log" 2>&1
rc=$?
echo "exit=$rc log=$log free_mib_at_start=$free"
if [ $rc -ne 0 ]; then
  tail -40 "$log"
else
  grep -vE 'INFO|WARNING|it/s|^Loading' "$log"
fi
