#!/usr/bin/env bash
# Like phase1_probe.sh, but runs against a pinned snapshot of the marathon package
# under ~/marathon-snap (populated by phase1_lengthsweep.sh) instead of the live repo
# checkout, so a concurrent edit in the worktree cannot break a running sweep.
set -uo pipefail
mkdir -p "$HOME/marathon-logs"
name="ls_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')"
log="$HOME/marathon-logs/$name.log"
cd "$HOME"
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH="$HOME/marathon-snap"
"$HOME/marathon-venv/bin/python" -m marathon.local_probe "$@" --json "$HOME/marathon-logs/$name.json" > "$log" 2>&1
rc=$?
echo "exit=$rc log=$log"
if [ $rc -ne 0 ]; then
  grep -n -iE 'error|memory|OOM|raise |failed|Exception' "$log" | grep -v 'File "' | head -20
else
  grep -h "copy_ms" "$log" | sed 's/.*shift:/  shift:/'
  grep -A80 -E '^ *turn ' "$log" | grep -vE 'INFO|WARNING|it/s'
fi
