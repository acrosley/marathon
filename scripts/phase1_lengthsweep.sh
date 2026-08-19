#!/usr/bin/env bash
# North-star sweep: edit-turn TTFT vs session length, prefix caching vs Marathon shift.
#
# For each context size the session is grown with the probe's ~593-token filler turns so
# the *edit turn* (always last-but-3) lands near the target prompt size; the edit rewrites
# turn 0 and the final turn asks the planted-fact parity question.
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_lengthsweep.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
snap="${MARATHON_SNAP:-}"
if [ -n "$snap" ]; then
  rm -rf "$HOME/marathon-snap"; mkdir -p "$HOME/marathon-snap"
  cp -r "$snap/marathon" "$HOME/marathon-snap/marathon"
  echo "pinned package snapshot -> $HOME/marathon-snap"
fi
run() { echo "=== $(date +%T) $* ==="; bash "$here/phase1_probe_pinned.sh" "$@"; }

# label  turns  gpu_util  store_tokens   (edit_at = turns-4)
sizes="
4k   10 0.80  8192
8k   17 0.80 12288
12k  24 0.80 16384
16k  30 0.80 20480
24k  44 0.78 27136
30k  54 0.78 33280
"

echo "$sizes" | while read -r label turns util store; do
  [ -z "${label:-}" ] && continue
  edit_at=$((turns - 4))
  common="--turns $turns --edit-at $edit_at --parity-tokens 16 --gpu-util $util"
  run --mode prefix $common
  run --mode shift  $common --store-tokens "$store"
done

# mid-history edits: the edited message sits in the middle of the session, not at turn 0
for spec in "16k 30 0.80 20480" "30k 54 0.78 33280"; do
  set -- $spec
  turns=$2; util=$3; store=$4
  edit_at=$((turns - 4))
  mid=$(( (turns - 4) / 2 ))
  common="--turns $turns --edit-at $edit_at --edit-turn $mid --parity-tokens 16 --gpu-util $util"
  run --mode prefix $common
  run --mode shift  $common --store-tokens "$store"
done
echo "=== $(date +%T) sweep done ==="
