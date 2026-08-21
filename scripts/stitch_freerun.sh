#!/usr/bin/env bash
# The discriminator: does the paged failure survive a *free-running* decode?
#
# The paged gate run measured 32 teacher-forced tokens and found planted-fact EM 498/500
# at a KL median of 0.0186 -- a big distributional failure with the answers intact. Track L
# measures free-running exact match on the real workload and sees fact EM 7/14. Exactly one
# of two things is true: teacher forcing was hiding an answer-level failure (a teacher-
# forced pass re-anchors to the reference every step, so divergence cannot compound), or
# the failure lives in the serving path rather than in stitched attention. Same population,
# same stitched cache, serving-shaped decode.
#
#   scripts/stitch_freerun.sh            # n=250 with the adapter column (~30 min)
#   scripts/stitch_freerun.sh --items 100 --no-lora
set -euo pipefail
REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
ADAPTER=$LOGS/stitch8b_w2.0.pt.step150
ITEMS=250
SEED=5101          # the same paged population the gate run measured
NAME=paged_freerun
LORA_ARG=()
while [ $# -gt 0 ]; do
  case "$1" in
    --items) ITEMS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --no-lora) ADAPTER=""; shift;;
    --name) NAME="$2"; shift 2;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done
[ -n "$ADAPTER" ] && LORA_ARG=(--lora "$ADAPTER")
mkdir -p "$LOGS"; cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m marathon.stitch_train freerun \
  --model Qwen/Qwen3-8B "${LORA_ARG[@]}" \
  --population paged --items "$ITEMS" --seed "$SEED" \
  --gen-tokens 32 --attn sdpa --jsonl "$LOGS/${NAME}.jsonl"
echo "FREERUN DONE -> $LOGS/${NAME}.jsonl"
