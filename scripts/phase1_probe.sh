#!/usr/bin/env bash
# Run local_probe inside WSL's marathon-venv, logging full output to ~/marathon-logs/.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_probe.sh --mode prefix --turns 16 --edit-at 13
set -uo pipefail
mkdir -p "$HOME/marathon-logs"
name="probe_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')"
log="$HOME/marathon-logs/$name.log"
cd "$HOME"
# ponytail: WSL2 has no UVA (pinned host-mapped memory); vLLM 0.27 v2 runner needs it
export VLLM_USE_V2_MODEL_RUNNER=0
"$HOME/marathon-venv/bin/python" -m marathon.local_probe "$@" --json "$HOME/marathon-logs/$name.json" > "$log" 2>&1
rc=$?
echo "exit=$rc log=$log"
if [ $rc -ne 0 ]; then
  grep -n -iE 'error|memory|OOM|raise |failed|Exception' "$log" | grep -v 'File "' | head -30
else
  grep -A40 -E '^ *turn ' "$log" | grep -vE 'INFO|WARNING|it/s'
fi
