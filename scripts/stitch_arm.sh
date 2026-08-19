#!/usr/bin/env bash
# One iteration-2 arm: train -> held-out eval (>=120, base and tuned on the same items)
# -> dependent-edit probe. Both arms run at 4-6k so they are comparable to each other;
# the base column is measured inside every eval, so no separate base-only run is needed.
#
#   scripts/stitch_arm.sh A            # hinge, cheap prefill
#   scripts/stitch_arm.sh B --grad-prefill
set -euo pipefail
ARM="$1"; shift || true

REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
NAME="stitch8b_arm${ARM}"
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# grad-prefill cap sits above the real ceiling (6k session + ~200 query tokens) so that in
# arm B *every* item keeps the fresh span in the graph and the arm measures one signal.
C=(--model Qwen/Qwen3-8B --gov-frac 0.5 --min-tokens 4000 --max-tokens 6000
   --gen-tokens 32 --attn sdpa --grad-prefill-max-tokens 8000)
run() { echo "+ $*"; "$@"; }

echo "=== arm $ARM 1/3: train (lr 3e-5, hinge 1.0, checkpoints every 50) ==="
run python -m marathon.stitch_train train "${C[@]}" --items 200 --seed 7001 --lr 3e-5 \
  --preserve-weight 1.0 --checkpoint-every 50 --mid-eval-items 16 "$@" \
  --out "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_train.jsonl"

echo "=== arm $ARM 2/3: held-out eval, n=120 (base and tuned, same items) ==="
run python -m marathon.stitch_train eval "${C[@]}" --items 120 --seed 9001 \
  --lora "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_eval.jsonl"

echo "=== arm $ARM 3/3: dependent-edit probe ==="
run python -m marathon.stitch_train probe --model Qwen/Qwen3-8B --gen-tokens 32 --attn sdpa \
  --lora "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_probe.jsonl"
echo "ARM $ARM DONE"
