#!/usr/bin/env bash
# Build LMCache 0.5.3 from source so its CUDA extension (lmcache.c_ops) links
# against the torch in marathon-venv.
#
# Why: the PyPI lmcache 0.5.3 wheel is built against an older libtorch, so its
# c_ops*.so fails to load on torch 2.13 (undefined symbol
# _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE). LMCache then
# silently falls back to its pure-torch ops, which are much slower and miss
# branches (see patch_lmcache_fused_kv.py). Rebuilding from source fixes both.
#
# Usage: wsl -d Ubuntu -- bash /mnt/c/Users/acros/Projects/marathon/scripts/phase1_build_lmcache.sh
set -euo pipefail
VENV="$HOME/marathon-venv"
REPO="/mnt/c/Users/acros/Projects/marathon"
SRC="$HOME/src/LMCache"
SITE="$VENV/lib/python3.12/site-packages"

# ponytail: no root in WSL (no apt, no system CUDA), but torch's own cu13 wheels
# already ship nvcc + headers + libs. Stitch them into a CUDA_HOME-shaped tree
# (bin/ include/ lib64/) which is all torch.utils.cpp_extension asks for.
CU="$SITE/nvidia/cu13"
export CUDA_HOME="$HOME/cuda-home"
rm -rf "$CUDA_HOME"; mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$CU/bin" "$CUDA_HOME/bin"
ln -sfn "$CU/include" "$CUDA_HOME/include"
ln -sfn "$CU/nvvm" "$CUDA_HOME/nvvm"
ln -sf "$CU"/lib/* "$CUDA_HOME/lib64/"
# the wheels ship only versioned sonames; the linker needs libfoo.so for -lfoo
(cd "$CUDA_HOME/lib64" && for f in *.so.*; do b="${f%%.so.*}"; [ -e "$b.so" ] || ln -sf "$f" "$b.so"; done)
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

[ -d "$SRC" ] || git clone --depth 1 --branch v0.5.3 https://github.com/LMCache/LMCache "$SRC"

# csrc/mem_kernels.cu::single_layer_kv_transfer (the op the layerwise CacheBlend
# path uses) has no branch for vLLM 0.27's fused/unified KV cache: it reads
# .size(4) on a 4-D tensor and its dispatch switch omits formats 12/13. Teach it
# the fused layout so blend runs on the native kernel instead of the pure-torch
# fallback. Idempotent: skip if already applied.
PATCH="$REPO/scripts/lmcache_fused_single_layer.patch"
if git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SRC" apply "$PATCH"
  echo "applied $PATCH"
else
  echo "fused single-layer patch already applied (or conflicts) -- skipping"
fi

export TORCH_CUDA_ARCH_LIST="12.0"   # Blackwell / RTX 5090
export MAX_JOBS=16
export LMCACHE_CUDA_MAJOR=13
# the wheels ship nvcc 13.3 next to a 13.0 cudart, and CCCL refuses that pairing
# outright. 13.0/13.3 are ABI-compatible in practice, so just skip the check.
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# --no-build-isolation: build against the venv's torch 2.13, not the torch 2.11
# pinned in LMCache's pyproject. --no-deps: never let it move torch/vllm.
uv pip install --python "$VENV/bin/python" setuptools wheel setuptools_scm ninja packaging
uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps \
  --reinstall-package lmcache "$SRC"

# the source install replaces the patched files, so re-apply (both idempotent)
"$VENV/bin/python" "$REPO/scripts/patch_vllm_blend.py"
"$VENV/bin/python" "$REPO/scripts/patch_lmcache_fused_kv.py"

# verify the extension is actually bound (lmcache/__init__.py shims a
# `lmcache.c_ops` module that forwards to the torch fallback, so a plain
# `import lmcache.c_ops` succeeds even when the compiled .so is broken)
"$VENV/bin/python" - <<'PYEOF'
import torch  # noqa: F401  (libtorch must be loaded first)
from lmcache.v1.platform.cuda.device_ops import CudaDeviceOps
ops = CudaDeviceOps()
ops.ensure_native()  # logs "compiled extension not found" if the .so failed
assert "pybind11" in repr(ops.multi_layer_kv_transfer), "c_ops NOT bound"
print("c_ops native OK")
PYEOF
