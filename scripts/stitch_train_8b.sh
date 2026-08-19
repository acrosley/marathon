#!/usr/bin/env bash
# Iteration 3: arm B's signal (--grad-prefill) with arm A's restraint (the do-no-harm
# hinge), at 4-8k, against the pre-registered gates. See docs/phase3-design.md, section
# "Iteration 3: pre-registration" -- the hinge weight, the checkpoint-selection rule, the
# regime and the standing-instruction bucket were all fixed before this was launched.
#
# Iteration 2 ran at 4-6k, where the *base* failure is 3.74x rather than the 7.25-8.71x
# measured at 4-8k, so its ratios flattered the gate. Step 1 therefore re-measures the base
# on this regime at n>=120 before an epoch is spent, exactly as the 8B gate run did.
#
#   scripts/stitch_train_8b.sh                       # w=2, the first arm
#   scripts/stitch_train_8b.sh --preserve-weight 4 --skip-basecheck   # w=4, if time allows
set -euo pipefail

REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
MODEL=Qwen/Qwen3-8B
CHECK_ITEMS=120   # >=120: ratios at n~50 are unstable (base moved 4.03x -> 2.27x between samples)
ITEMS=200
EVAL_ITEMS=120
TRAIN_SEED=7001
EVAL_SEED=9001
GOV_FRAC=0.5
STANDING=0.34     # share of the governing half drawn as probe-shaped standing-instruction
LR=3e-5           # 1e-4 spiked clean_kl to 0.0223 by step 81 at 0.6B; back it off
PRESERVE=2.0      # hinge weight: {1,2,4} pre-registered, w=1 is iteration 2, w=2 runs first
CKPT=50           # mid-training checkpoint + held-out eval every N items
MID_ITEMS=24
MIN_TOK=4000
MAX_TOK=8000
# above the real ceiling (8k session + query), so no item is capped out of the expressive
# path by prediction; an item that genuinely does not fit OOMs, falls back and is counted
GP_CAP=8600
SKIP_BASE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --check-items) CHECK_ITEMS="$2"; shift 2;;
    --items) ITEMS="$2"; shift 2;;
    --eval-items) EVAL_ITEMS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --preserve-weight) PRESERVE="$2"; shift 2;;
    --standing-frac) STANDING="$2"; shift 2;;
    --grad-prefill-max-tokens) GP_CAP="$2"; shift 2;;
    --min-tokens) MIN_TOK="$2"; shift 2;;
    --max-tokens) MAX_TOK="$2"; shift 2;;
    --skip-basecheck) SKIP_BASE=1; shift;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done

NAME="stitch8b_w${PRESERVE}"

mkdir -p "$LOGS"
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(--model "$MODEL" --gov-frac "$GOV_FRAC" --standing-frac "$STANDING"
        --min-tokens "$MIN_TOK" --max-tokens "$MAX_TOK" --gen-tokens 32 --attn sdpa
        --grad-prefill-max-tokens "$GP_CAP")
run() { echo "+ $*"; "$@"; }

if [ -z "$SKIP_BASE" ]; then
  echo "=== step 1/4: base-only, n=$CHECK_ITEMS at ${MIN_TOK}-${MAX_TOK} (the ratio to beat) ==="
  run python -m marathon.stitch_train eval "${COMMON[@]}" --items "$CHECK_ITEMS" \
    --seed "$EVAL_SEED" --base-only --jsonl "$LOGS/stitch8b_it3_basecheck.jsonl"
fi

echo "=== step 2/4: train w=$PRESERVE (grad-prefill, hinge, checkpoints every $CKPT) ==="
run python -m marathon.stitch_train train "${COMMON[@]}" --items "$ITEMS" \
  --seed "$TRAIN_SEED" --lr "$LR" --preserve-weight "$PRESERVE" --grad-prefill \
  --checkpoint-every "$CKPT" --mid-eval-items "$MID_ITEMS" \
  --out "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_train.jsonl"

# The pre-registered rule (best governing p95 subject to non-gov median <= base x1.2 and
# clean <= 0.002) is printed by the trainer and its pick is written next to the adapter.
# Evaluate that checkpoint when it selected one, the final adapter when it did not.
SEL=$(python - "$LOGS/${NAME}_train.jsonl.checkpoints" <<'PY'
import json, sys
from marathon.stitch_train import select_checkpoint
hist = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
pick = select_checkpoint(hist)
print(pick["path"] if pick and pick.get("path") else "")
PY
)
ADAPTER="$LOGS/${NAME}.pt"
if [ -n "$SEL" ] && [ -f "$SEL" ]; then
  echo "selected checkpoint: $SEL"
  ADAPTER="$SEL"
else
  echo "no checkpoint feasible under the rule -- evaluating the final adapter"
fi

echo "=== step 3/4: held-out eval, n=$EVAL_ITEMS, same items as step 1 ==="
run python -m marathon.stitch_train eval "${COMMON[@]}" --items "$EVAL_ITEMS" \
  --seed "$EVAL_SEED" --lora "$ADAPTER" --jsonl "$LOGS/${NAME}_eval.jsonl"

echo "=== step 4/4: dependent-edit probe (still held out: different builder, 20 turns) ==="
run python -m marathon.stitch_train probe --model "$MODEL" --gen-tokens 32 --attn sdpa \
  --lora "$ADAPTER" --jsonl "$LOGS/${NAME}_probe.jsonl"
echo "ARM w=$PRESERVE DONE (adapter $ADAPTER)"
