"""Teach LMCache's pure-torch KV fallback about vLLM's fused (unified) KV cache.

Why this exists
---------------
The lmcache 0.5.3 PyPI wheel ships a compiled ``lmcache/c_ops*.so`` built
against an older libtorch. Against torch 2.13 it fails to load with::

    undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE

LMCache catches that ImportError and silently falls back to the pure-torch
baseline in ``lmcache/v1/platform/torch_ops.py`` (the "c_ops compiled extension
not found" warning). That baseline is incomplete: ``single_layer_kv_transfer``
-- the op the layerwise CacheBlend path uses -- only handles MLA and the older
5-D per-layer layouts. It has no branch for the fused K/V formats (enum 10-13),
so it indexes ``.size(4)`` on vLLM 0.27's 4-D cache and dies with
``IndexError: Dimension out of range``.

vLLM 0.27.1 FlashAttention packs K and V into the trailing dim
(``flash_attn.py: get_kv_cache_shape -> (num_blocks, num_kv_heads, block_size,
2 * head_size)``; NHD stride order hands LMCache
``[num_blocks, block_size, num_kv_heads, 2 * head_size]``, detected as
``EngineKVFormat.NL_X_NB_BS_NH_CS`` == 13). K is the first ``head_size`` slice
of the content dim, V the second -- see vLLM's own
``kv_cache.transpose(1, 2).split(head_size, dim=-1)``.

This patch adds the missing fused branch, mirroring the conventions of the
neighbouring ``_transfer_per_layer_fused`` (multi-layer path) so the two agree.

scripts/phase1_build_lmcache.sh rebuilds c_ops from source *and* applies
scripts/lmcache_fused_single_layer.patch, which adds the same fused branch to
csrc/mem_kernels.cu -- so in practice the native kernel runs and this torch
branch is only the fallback for when the .so fails to load.

Idempotent; run with the venv python. Drop once upstream LMCache handles the
fused K/V formats in single_layer_kv_transfer.
"""

import pathlib

import lmcache.v1.platform.torch_ops as t

MARKER = "marathon: fused K/V"

ANCHOR = """    if is_mla:
        # ── MLA format ──
"""

INJECT = """    if _is_fused_kv_format(engine_kv_format):
        # marathon: fused K/V per-layer formats (vLLM 0.27 unified KV cache).
        # vllm NHD (fmt 11/13): [num_blocks, block_size, num_heads, 2 * head_size]
        # vllm HND (fmt 10/12): [num_blocks, num_heads, block_size, 2 * head_size]
        # K is content[:head_size], V is content[head_size:].
        # lmc: [2, num_tokens, num_heads * head_size] (or token-major).
        is_hnd_fused = int(engine_kv_format) in (
            int(EngineKVFormat.NL_X_NB_NH_BS_TWO_HS),
            int(EngineKVFormat.NL_X_NB_NH_BS_CS),
        )
        paged = vllm_key_value_cache
        if paged.dim() == 5:
            # Canonical [..., 2, head_size] split view; flatten the pair back.
            paged = paged.reshape(*paged.shape[:3], -1)
        if is_hnd_fused:
            num_heads, block_size = paged.size(1), paged.size(2)
            paged = paged.permute(0, 2, 1, 3)
        else:
            block_size, num_heads = paged.size(1), paged.size(2)
        head_size = paged.size(3) // 2
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        if int(direction) == int(TransferDirection.D2H):
            gathered = paged[block_indices, block_offsets]
            gathered = gathered.reshape(-1, num_heads, 2, head_size)
            for kv in range(2):
                flat = gathered[:, :, kv, :].reshape(-1, num_heads * head_size)
                flat = flat.to(lmc_key_value_cache.device)
                if token_major:
                    lmc_key_value_cache[valid_token_indices, kv] = flat
                else:
                    lmc_key_value_cache[kv, valid_token_indices] = flat
        else:
            planes = []
            for kv in range(2):
                if token_major:
                    lmc_src = lmc_key_value_cache[valid_token_indices, kv]
                else:
                    lmc_src = lmc_key_value_cache[kv, valid_token_indices]
                planes.append(
                    lmc_src.reshape(-1, num_heads, head_size).to(paged_memory_device)
                )
            paged[block_indices, block_offsets] = torch.cat(planes, dim=-1)
        return

"""


# The compiled c_ops used to have the same gap for this one op, so we unbound
# it here and fell back to torch. It no longer does -- scripts/phase1_probe.sh's
# build applies scripts/lmcache_fused_single_layer.patch, which adds the fused
# branch to csrc/mem_kernels.cu -- so the native binding stays and the torch
# branch above is only a safety net (used when the .so fails to load).


def _patch(path: pathlib.Path, anchor: str, replacement: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print("already patched", path)
        return
    assert s.count(anchor) == 1, f"{path.name} changed; re-inspect the patch"
    path.write_text(s.replace(anchor, replacement, 1))
    print("patched", path)


def main() -> None:
    _patch(pathlib.Path(t.__file__), ANCHOR, INJECT + ANCHOR, MARKER)


if __name__ == "__main__":
    main()
