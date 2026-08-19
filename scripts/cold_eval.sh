#!/usr/bin/env bash
# Phase 2 cold-tier eval. Usage: cold_eval.sh [extra args to marathon.cold_eval]
set -euo pipefail
REPO=/mnt/c/Users/acros/Projects/marathon-phase2
LOGS=~/marathon-logs
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d-%H%M%S)
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src"
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
cd "$REPO"
python -m marathon.cold_eval --out "$LOGS/cold_eval-$STAMP.jsonl" "$@" \
  2>&1 | tee "$LOGS/cold_eval-$STAMP.log"
