#!/usr/bin/env bash
# Session-keyed connector check: small-model parity, two interleaved sessions, and the
# 14B regression run that the scheduler-safe rewrite must not slow down.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_sessions.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
run() { echo "=== $* ==="; bash "$here/phase1_probe.sh" "$@"; }

small="--model Qwen/Qwen3-0.6B --max-model-len 16384 --parity-tokens 16"

# (a) does --mode shift still reproduce prefix mode on one session?
run --mode prefix $small --turns 12 --edit-at 9
run --mode shift  $small --turns 12 --edit-at 9 --store-tokens 16384

# (b) two independent sessions, interleaved, both edited, different planted facts
run --mode prefix $small --turns 12 --edit-at 9 --sessions 2
run --mode shift  $small --turns 12 --edit-at 9 --sessions 2 --store-tokens 24576

# (c) the standard 14B run: edit-turn time must still be ~0.2-0.25 s
run --mode shift --turns 24 --edit-at 20 --parity-tokens 16 --store-tokens 16384
