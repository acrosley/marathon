#!/usr/bin/env bash
# Stitched-KV consistency fine-tuning pilot: baseline eval -> train -> post eval.
#
#   scripts/stitch_train.sh [--model Qwen/Qwen3-0.6B] [--items N] [--eval-items N] ...
#
# Everything after the known flags is passed through to `python -m marathon.stitch_train`.
# Train and eval populations are different seeds by construction (TRAIN_SEED/EVAL_SEED):
# the eval seed is never trained on.
set -euo pipefail

REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
MODEL=Qwen/Qwen3-0.6B
ITEMS=600
EVAL_ITEMS=84
TRAIN_SEED=7001
EVAL_SEED=9001
GOV_FRAC=0.5
TAG=""
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --items) ITEMS="$2"; shift 2;;
    --eval-items) EVAL_ITEMS="$2"; shift 2;;
    --train-seed) TRAIN_SEED="$2"; shift 2;;
    --eval-seed) EVAL_SEED="$2"; shift 2;;
    --gov-frac) GOV_FRAC="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

[ "$TRAIN_SEED" != "$EVAL_SEED" ] || { echo "train and eval seeds must differ"; exit 2; }

NAME="stitch_$(basename "$MODEL")${TAG:+_$TAG}"
mkdir -p "$LOGS"
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src"
# session lengths vary 3-6k, so the big KV buffers are all different sizes; without
# expandable segments the allocator fragments and the run degrades into a crawl.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "+ $*"; "$@"; }

# No separate baseline run: `eval` reports base and tuned side by side on the same
# examples (adapters off is the base model, bit for bit), so one post-tune eval is
# the whole before/after table and the two columns cannot drift apart on sampling.
echo "=== $NAME: train (seed $TRAIN_SEED) ==="
run python -m marathon.stitch_train train --model "$MODEL" --items "$ITEMS" \
  --seed "$TRAIN_SEED" --gov-frac "$GOV_FRAC" --out "$LOGS/${NAME}.pt" \
  --jsonl "$LOGS/${NAME}_train.jsonl" "${EXTRA[@]+"${EXTRA[@]}"}"

echo "=== $NAME: post-tune eval (same held-out seed) ==="
run python -m marathon.stitch_train eval --model "$MODEL" --items "$EVAL_ITEMS" \
  --seed "$EVAL_SEED" --gov-frac "$GOV_FRAC" --lora "$LOGS/${NAME}.pt" \
  --jsonl "$LOGS/${NAME}_eval_tuned.jsonl" "${EXTRA[@]+"${EXTRA[@]}"}"

echo "=== $NAME: dependent-edit probe (kvshift_probe scenarios, base vs tuned) ==="
run python -m marathon.stitch_train probe --model "$MODEL" --lora "$LOGS/${NAME}.pt" \
  --jsonl "$LOGS/${NAME}_probe.jsonl" "${EXTRA[@]+"${EXTRA[@]}"}"
