"""Register vLLM's model with LMCache's tracker so CacheBlend can build its blender.

LMCache examples/blend_kv_v1/README calls this an "ad-hoc change needed in vLLM";
nothing in lmcache 0.5.3 does it. Patches the installed vllm/v1/worker/gpu_worker.py
in place, idempotently. Run with the venv python. Revisit when LMCache upstreams it.
"""

import pathlib

import vllm.v1.worker.gpu_worker as w

p = pathlib.Path(w.__file__)
s = p.read_text()
anchor = "        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)\n"
inject = (
    "        # marathon: register model for LMCache CacheBlend (blend needs the nn.Module)\n"
    "        try:\n"
    "            from lmcache.integration.vllm.utils import ENGINE_NAME\n"
    "            from lmcache.v1.compute.models.utils import VLLMModelTracker\n"
    "\n"
    "            VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\n"
    "        except ImportError:\n"
    "            pass\n"
)
if "marathon: register model" in s:
    print("already patched", p)
else:
    assert s.count(anchor) == 1, "gpu_worker.py changed; re-inspect the patch"
    p.write_text(s.replace(anchor, inject + anchor))
    print("patched", p)
