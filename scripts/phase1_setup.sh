#!/usr/bin/env bash
# Phase 1 environment: vLLM + LMCache on cu13 wheels, in WSL2 Ubuntu.
# Usage (from Windows): wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_setup.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV="$HOME/marathon-venv"
REPO="/mnt/c/Users/acros/Projects/marathon"
MODEL="${MODEL:-Qwen/Qwen3-14B-FP8}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
# ponytail: pinned to what we measured with; bump deliberately and log in findings.md
uv pip install --python "$VENV/bin/python" \
  vllm==0.27.1 lmcache==0.5.3 "huggingface_hub[hf_transfer]"
# ponytail: prebuilt FlashInfer kernels; no nvcc in WSL, and JIT would need CUDA toolkit
uv pip install --python "$VENV/bin/python" \
  flashinfer-jit-cache==0.6.16.post3 --extra-index-url https://flashinfer.ai/whl/cu130
uv pip install --python "$VENV/bin/python" -e "$REPO"

# the PyPI lmcache wheel's c_ops .so does not link against torch 2.13, so LMCache
# silently falls back to slow pure-torch ops. Rebuild it from source; this also
# (re-)applies both patches below.
bash "$REPO/scripts/phase1_build_lmcache.sh"

"$VENV/bin/python" - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("vllm", vllm.__version__)
PY

HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/hf" download "$MODEL"
echo "setup done: $MODEL cached, marathon installed"
