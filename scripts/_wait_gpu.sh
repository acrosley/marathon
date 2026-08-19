#!/usr/bin/env bash
# Block until the GPU is genuinely free of another tenant, then exit 0.
# Runs entirely inside WSL so there is no wsl.exe quoting/stderr noise to misparse, and
# `pgrep -f` is given a bracketed pattern so it cannot match this script's own command line.
need=${1:-3}      # consecutive clear polls required
gap=${2:-180}
ok=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  procs=$(pgrep -fc '[c]old_eval|[E]ngineCore|[v]llm' 2>/dev/null || true)
  procs=${procs:-0}
  if [ -n "$used" ] && [ "$used" -lt 6000 ] && [ "$procs" -eq 0 ]; then
    ok=$((ok + 1))
  else
    ok=0
  fi
  echo "$(date +%H:%M:%S) used=${used:-?}MiB other_procs=$procs clear=$ok/$need"
  [ "$ok" -ge "$need" ] && break
  sleep "$gap"
done
echo "GPU FREE"
