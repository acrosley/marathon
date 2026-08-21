#!/usr/bin/env bash
# Is the paged answer-level collapse in the connector, or in the model?
#
# Phase 3 measured a fully stitched HF cache on Qwen3-8B over the paged population and
# got planted-fact EM 250/250 -- stale attention does not lose facts. The serving path on
# Qwen3-14B-FP8 gets 7/14 on the same workload shape. Same workload, same model family,
# two serving paths. So: run the *same* 8B, bf16, through vLLM + the shift connector.
#   collapses  -> the connector/serving path
#   holds      -> FP8/14B-specific
#
# Run 1 also answers "is the stitched KV itself right?" inside a live engine:
# MARATHON_VERIFY_LOAD reads every loaded span back out of the paged cache and diffs it
# against kvshift.rerotate_keys on the same source rows, per layer.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$HOME/marathon-logs"; mkdir -p "$out"

paged=(--demote 0 --max-tokens 16 --fixed-replies --fact-probe)

echo "########## 1. in-engine stitched-KV verification (0.6B, 12 turns)"
bash "$here/server_demo.sh" --model Qwen/Qwen3-0.6B --gpu-util 0.30 --max-model-len 16384 \
  --store-tokens 16384 --turns 16 --active-window 2048 "${paged[@]}" --max-churn 1.0 --verify-load \
  --json "$out/disc_verify.json"
sleep 20

echo "########## 2. Qwen3-8B bf16, connector ON (max_churn 1.0)"
bash "$here/server_demo.sh" --model Qwen/Qwen3-8B --gpu-util 0.85 --max-model-len 40960 \
  --store-tokens 16384 --turns 40 "${paged[@]}" --max-churn 1.0 \
  --json "$out/disc_8b_reuse.json"
sleep 20

echo "########## 3. Qwen3-8B bf16, connector OFF (control)"
bash "$here/server_demo.sh" --model Qwen/Qwen3-8B --gpu-util 0.85 --max-model-len 40960 \
  --store-tokens 16384 --turns 40 "${paged[@]}" --no-reuse \
  --json "$out/disc_8b_control.json"
echo "done"
