#!/usr/bin/env bash
set -euo pipefail
mkdir -p ~/marathon-logs
~/marathon-venv/bin/python - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json","*.safetensors","*.txt","*.py"])
print(p)
EOF
