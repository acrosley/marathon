#!/usr/bin/env bash
# Block until PATTERN appears in FILE or PROC stops running. Exists because loop variables
# do not survive the PowerShell/Git-Bash -> wsl.exe -> bash quoting chain.
#   _wait_file.sh <file> <pattern> <procpattern> [max_polls] [gap]
FILE="$1"; PAT="$2"; PROC="$3"; MAX="${4:-90}"; GAP="${5:-10}"
n=0
while [ "$n" -lt "$MAX" ]; do
  if grep -qE "$PAT" "$FILE" 2>/dev/null; then break; fi
  if ! pgrep -f "$PROC" > /dev/null 2>&1; then echo "[process gone]"; break; fi
  n=$((n + 1)); sleep "$GAP"
done
