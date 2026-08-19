#!/usr/bin/env bash
# Run the position-shifted KV reuse probe inside WSL's marathon-venv.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/kvshift_probe.sh --model Qwen/Qwen3-0.6B
set -uo pipefail
mkdir -p "$HOME/marathon-logs"
name="kvshift_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')"
log="$HOME/marathon-logs/$name.log"
cd "$HOME"
# refuse to start if the GPU is busy (another agent runs vLLM on the same card)
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
need=${KVSHIFT_NEED_MIB:-4000}
if [ "$free" -lt "$need" ]; then
  echo "GPU has only ${free} MiB free (< ${need}); not starting."
  exit 75
fi
"$HOME/marathon-venv/bin/python" -m marathon.kvshift_probe "$@" \
  --json "$HOME/marathon-logs/$name.json" > "$log" 2>&1
rc=$?
echo "exit=$rc log=$log free_mib_at_start=$free"
if [ $rc -ne 0 ]; then
  tail -40 "$log"
else
  grep -vE 'INFO|WARNING|it/s|^Loading' "$log"
fi
