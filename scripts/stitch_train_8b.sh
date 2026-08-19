#!/usr/bin/env bash
# The 8B pilot, in the regime where the failure class actually exists.
#
# The 0.6B pilot (findings.md 2026-08-19) was negative for a reason that had nothing to do
# with the method: at 0.6B and 3-5k tokens a governing edit costs 1.50x a non-governing one,
# not the ~9x the 144-session Qwen3-8B eval measured, so there was nothing to repair. This
# runs the same experiment at 8B on 4-8k sessions, and -- the point of step 1 -- refuses to
# spend an epoch of training until the base model is shown to exhibit the failure.
#
#   scripts/stitch_train_8b.sh [--check-items N] [--items N] [--eval-items N] [--lr X]
set -euo pipefail

REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
MODEL=Qwen/Qwen3-8B
CHECK_ITEMS=48
ITEMS=200
EVAL_ITEMS=60
TRAIN_SEED=7001
EVAL_SEED=9001
GOV_FRAC=0.5
LR=3e-5           # 1e-4 spiked clean_kl to 0.0223 by step 81 at 0.6B; back it off
MIN_TOK=4000
MAX_TOK=8000
NAME=stitch_8b

while [ $# -gt 0 ]; do
  case "$1" in
    --check-items) CHECK_ITEMS="$2"; shift 2;;
    --items) ITEMS="$2"; shift 2;;
    --eval-items) EVAL_ITEMS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --min-tokens) MIN_TOK="$2"; shift 2;;
    --max-tokens) MAX_TOK="$2"; shift 2;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done

mkdir -p "$LOGS"
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(--model "$MODEL" --gov-frac "$GOV_FRAC" --min-tokens "$MIN_TOK" --max-tokens "$MAX_TOK" --gen-tokens 32 --attn sdpa)
run() { echo "+ $*"; "$@"; }

echo "=== step 1/4: does the failure class reproduce in the BASE model? (held-out seed) ==="
run python -m marathon.stitch_train eval "${COMMON[@]}" --items "$CHECK_ITEMS" \
  --seed "$EVAL_SEED" --base-only --jsonl "$LOGS/${NAME}_basecheck.jsonl"

echo "=== step 2/4: train (seed $TRAIN_SEED, lr $LR) ==="
run python -m marathon.stitch_train train "${COMMON[@]}" --items "$ITEMS" \
  --seed "$TRAIN_SEED" --lr "$LR" --out "$LOGS/${NAME}.pt" \
  --jsonl "$LOGS/${NAME}_train.jsonl"

echo "=== step 3/4: held-out eval, base vs tuned vs clean drift ==="
run python -m marathon.stitch_train eval "${COMMON[@]}" --items "$EVAL_ITEMS" \
  --seed "$EVAL_SEED" --lora "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_eval.jsonl"

echo "=== step 4/4: dependent-edit probe ==="
run python -m marathon.stitch_train probe --model "$MODEL" --gen-tokens 32 --attn sdpa \
  --lora "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_probe.jsonl"
