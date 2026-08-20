#!/usr/bin/env bash
# The gate run: eval-only, under the 2026-08-20 protocol. No training.
#
# Iterations 1-3 each trained an adapter and then measured it with a statistic that could
# not resolve the effect: at n=46 stable governing items the test had 18% power, a quarter
# of items had an undetermined reference, and the whole apparent governing win was one
# session. So this run trains nothing. It re-measures the *existing* w=2 adapter
# (checkpoint 150, the one the pre-registered rule selected) on enough items, with the
# reference-stability probe on, and reports the paired per-item deltas with bootstrap
# intervals. See docs/phase3-design.md, "The measurement rebuild".
#
#   scripts/stitch_gate_eval.sh                 # the full n=1400 gate run (~2.0 h)
#   scripts/stitch_gate_eval.sh --items 800     # the short fallback (~1.1 h)
set -euo pipefail

REPO=/mnt/c/Users/acros/Projects/marathon-phase3
LOGS=~/marathon-logs
MODEL=Qwen/Qwen3-8B
ADAPTER=$LOGS/stitch8b_w2.0.pt.step150
ITEMS=1400
# Sized for >=400 reference-stable items in BOTH gated buckets. At n=1400 with these
# fractions the split is 641 core-governing / 126 standing / 633 non-governing; at the ~74%
# stable rate seen on iteration 3's data that is 474 / 93 / 468. The margin over 400 is
# deliberate -- the stable rate under the *new* probe has never been measured, and at a
# pessimistic 65% this still clears the bar (417 / 411).
GOV_FRAC=0.55
STANDING=0.18
# NOT 9001. The iteration-3 training run drew its mid-training slice from --seed+2000 =
# 9001, which was also the eval seed, so 24 of those 120 held-out items were used to select
# the checkpoint being tested here. 9101 was never trained on and never selected on.
SEED=9101
MIN_TOK=4000
MAX_TOK=8000
NAME=stitch8b_gate

while [ $# -gt 0 ]; do
  case "$1" in
    --items) ITEMS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --lora) ADAPTER="$2"; shift 2;;
    --gov-frac) GOV_FRAC="$2"; shift 2;;
    --standing-frac) STANDING="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done

[ -f "$ADAPTER" ] || { echo "adapter not found: $ADAPTER"; exit 1; }

mkdir -p "$LOGS"
cd "$REPO"
# shellcheck disable=SC1090
source ~/marathon-venv/bin/activate
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== gate eval: $ITEMS items, seed $SEED, adapter $ADAPTER ==="
echo "    rows stream to $LOGS/${NAME}_eval.jsonl -- a run stopped at the 17:30 hard stop"
echo "    is still analysable, and 1000 items already buys ~99%/80% power."
# --ref-stability: one extra prefill + one teacher-forced pass per item (~17%), which is
# what makes the gated statistic well posed. No --grad-prefill: nothing is being trained.
python -m marathon.stitch_train eval \
  --model "$MODEL" --lora "$ADAPTER" --ref-stability \
  --items "$ITEMS" --seed "$SEED" --gov-frac "$GOV_FRAC" --standing-frac "$STANDING" \
  --min-tokens "$MIN_TOK" --max-tokens "$MAX_TOK" --gen-tokens 32 --attn sdpa \
  --jsonl "$LOGS/${NAME}_eval.jsonl"
echo "GATE EVAL DONE -> $LOGS/${NAME}_eval.jsonl"
