"""Let a KV connector declare non-prefix external matches, so k segments load in ONE request.

vLLM's connector API expresses externally supplied KV only as a *prefix*:
``get_num_new_matched_tokens`` returns a count and the scheduler folds it into a scalar
``num_computed_tokens``, after which the model runner prefills the contiguous suffix
``[num_computed, num_computed + num_scheduled)``. That contract is why
``marathon.reuse_plan.phases`` hands ``k`` reused segments over as ``k + 1`` sequential
requests -- and measurement on 2026-08-21 put the paged workload's answer-level collapse
in that multi-request path, not in the stitched KV, which is bit-right to bf16 in every
layer.

Prefilling only the *gaps* between reused spans is sound, and for one reason: attention
reads earlier positions out of the paged cache through the block table, so a token has to
be *present*, not *computed in this pass*. The connector writes reused spans in
``start_load_kv``, before the forward; gaps earlier in the same batch are written to their
slots before later gaps attend to them.

Two surgical edits, both idempotent, both reverted by ``--revert``:

1. ``v1/core/sched/scheduler.py`` -- after the external-match count is folded in, ask the
   connector for an explicit gap list. If it supplies one, publish ``(positions, matched)``
   for the runner and let ``num_computed_tokens`` mean "matched + gaps already computed",
   which keeps every existing "is this request done prefilling" test correct.
2. ``v1/worker/gpu_model_runner.py`` -- ``positions_np`` is the single place the
   contiguous-suffix assumption enters input construction. For a request with a gap list,
   overwrite its slice with the explicit positions. Everything downstream (token ids,
   slot mapping, block tables, attention) is derived from ``positions_np``, so nothing
   else has to change.

The channel between them is ``marathon.gapfill_channel``, a module-level dict; the v1
scheduler and worker share a process on a single GPU. Tensor parallelism is not
supported and not attempted.

NOT YET RUN ON HARDWARE -- written 2026-08-21 with no GPU available. Anchors are asserted
exactly once each, so a vLLM upgrade fails loudly here rather than silently mis-patching.

    python scripts/patch_vllm_gapfill.py            # apply
    python scripts/patch_vllm_gapfill.py --revert   # restore
"""

from __future__ import annotations

import argparse
import pathlib
import sys

MARK = "# marathon: gapfill"
SUFFIX = ".marathon-gapfill.bak"

SCHED_ANCHOR = """                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
"""

SCHED_INJECT = f"""                    {MARK}: a connector may declare a non-prefix match. It
                    # publishes the positions the engine must still compute; keeping
                    # num_computed_tokens = matched + computed-gaps leaves every
                    # "done prefilling" comparison in this file correct.
                    try:
                        from marathon import gapfill_channel

                        _gap = gapfill_channel.take(request.request_id)
                    except Exception:
                        _gap = None
                    if _gap is not None:
                        _positions, _matched = _gap
                        num_computed_tokens = _matched
                        gapfill_channel.publish(request.request_id, _positions, _matched)
"""

RUNNER_ANCHOR = """        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )
"""

RUNNER_INJECT = f"""        {MARK}: a request with a non-prefix match computes an explicit set of
        # positions, not a contiguous suffix. positions_np is the only place the
        # contiguity assumption enters input construction, so overwriting this slice is
        # the whole change -- token ids, slot mapping and block tables all derive from it.
        try:
            from marathon import gapfill_channel

            _gaps = gapfill_channel.active()
        except Exception:
            _gaps = None
        if _gaps:
            for _i in range(num_reqs):
                _g = _gaps.get(self.input_batch.req_ids[_i])
                if _g is None:
                    continue
                _positions, _matched = _g
                _lo = int(cu_num_tokens[_i - 1]) if _i else 0
                _hi = int(cu_num_tokens[_i])
                _c = int(self.input_batch.num_computed_tokens_cpu[_i]) - _matched
                positions_np[_lo:_hi] = _positions[_c : _c + (_hi - _lo)]
"""

EDITS = [
    ("vllm/v1/core/sched/scheduler.py", SCHED_ANCHOR, SCHED_INJECT, "after"),
    ("vllm/v1/worker/gpu_model_runner.py", RUNNER_ANCHOR, RUNNER_INJECT, "after"),
]


def _root() -> pathlib.Path:
    import vllm

    return pathlib.Path(vllm.__file__).parent.parent


def apply(revert: bool = False) -> int:
    root = _root()
    for rel, anchor, inject, where in EDITS:
        path = root / rel
        backup = path.with_suffix(path.suffix + SUFFIX)
        if revert:
            if backup.exists():
                path.write_text(backup.read_text())
                backup.unlink()
                print("reverted", path)
            else:
                print("no backup, skipping", path)
            continue
        text = path.read_text()
        if MARK in text:
            print("already patched", path)
            continue
        count = text.count(anchor)
        if count != 1:
            print(f"ERROR: anchor found {count}x in {path}; vLLM changed, re-inspect")
            return 1
        backup.write_text(text)
        new = inject + anchor if where == "before" else anchor + inject
        path.write_text(text.replace(anchor, new))
        print("patched", path, f"(backup {backup.name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true", help="restore the backed-up files")
    return apply(ap.parse_args(argv).revert)


if __name__ == "__main__":
    sys.exit(main())
