#!/usr/bin/env bash
# The paged gate run: the powered protocol, on Track L's workload shape.
#
# One pass, not two: `evaluate` computes the base and tuned columns against the *same*
# teacher for each item, which is exactly what the paired-delta protocol requires. Running
# base and adapter separately would reintroduce the cross-run comparison that iteration 3
# proved is not measurable.
#
#   scripts/stitch_paged_gate.sh                 # n=500 (~45 min)
#   scripts/stitch_paged_gate.sh --items 250
set -euo pipefail
REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
ADAPTER=$LOGS/stitch8b_w2.0.pt.step150
ITEMS=500
SEED=5101          # fresh: never trained on, never selected on
NAME=paged_gate
while [ $# -gt 0 ]; do
  case "$1" in
    --items) ITEMS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --lora) ADAPTER="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done
[ -f "$ADAPTER" ] || { echo "adapter not found: $ADAPTER"; exit 1; }
mkdir -p "$LOGS"; cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m marathon.stitch_train eval \
  --model Qwen/Qwen3-8B --lora "$ADAPTER" --ref-stability \
  --population paged --items "$ITEMS" --seed "$SEED" \
  --gen-tokens 32 --attn sdpa --jsonl "$LOGS/${NAME}_eval.jsonl"
echo "PAGED GATE DONE -> $LOGS/${NAME}_eval.jsonl"
