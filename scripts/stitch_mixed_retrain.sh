#!/usr/bin/env bash
# Clean retrain on the mixed population, then eval on the paged one.
#
# The paged gate run showed the w=2 adapter transfers *nothing* to the workload shape that
# matters (paired delta -0.00013, CI straddling zero) while that population's base failure
# is far worse than the synthetic one (median 0.0186 against 0.0021/0.0071). So this trains
# on both shapes -- paged items are TARGETS, not hinge-guarded -- and measures on paged.
set -euo pipefail
REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
NAME=stitch8b_mixed
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=(--model Qwen/Qwen3-8B --gen-tokens 32 --attn sdpa --gov-frac 0.55 --standing-frac 0.18
   --min-tokens 4000 --max-tokens 8000 --grad-prefill-max-tokens 6000)

echo "=== 1/2: train on mixed (paged + synthetic), w=2, seed 7101 ==="
python -m marathon.stitch_train train "${C[@]}" --population mixed \
  --items "${ITEMS:-200}" --seed 7101 --lr 3e-5 --preserve-weight 2.0 --grad-prefill \
  --checkpoint-every 50 --mid-eval-items 16 --mid-eval-seed 7999 \
  --out "$LOGS/${NAME}.pt" --jsonl "$LOGS/${NAME}_train.jsonl"

SEL=$(python - "$LOGS/${NAME}_train.jsonl.checkpoints" <<'PY'
import json, sys
from marathon.stitch_train import select_checkpoint
h=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
p=select_checkpoint(h); print(p["path"] if p and p.get("path") else "")
PY
)
ADAPTER="$LOGS/${NAME}.pt"
[ -n "$SEL" ] && [ -f "$SEL" ] && ADAPTER="$SEL" && echo "selected checkpoint: $SEL"

echo "=== 2/2: eval on the PAGED population, same items as the gate run (seed 5101) ==="
python -m marathon.stitch_train eval --model Qwen/Qwen3-8B --lora "$ADAPTER" \
  --ref-stability --population paged --items "${EVAL_ITEMS:-250}" --seed 5101 \
  --gen-tokens 32 --attn sdpa --jsonl "$LOGS/${NAME}_paged_eval.jsonl"
echo "MIXED RETRAIN DONE (adapter $ADAPTER)"
