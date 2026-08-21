"""Measure the real per-item peak on the mixed population before committing a run.

Yesterday's mixed retrain died in the backward pass inside an hour-long job that had
produced nothing, because the grad-prefill cap was chosen from an estimate. This runs the
actual training step over a handful of items, printing the measured peak per item and the
token length that produced it, so the cap is picked from a measurement.

    python scripts/stitch_sizing.py --items 20 --cap 6000
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7101)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no-grad-prefill", action="store_true")
    a = ap.parse_args()

    from marathon.paged_eval import build_paged_examples
    from marathon.stitch_train import _load, build_examples, example_losses

    tok, model, loras = _load(a.model, "cuda", "sdpa", 16, 32, 0.0)
    dev = next(model.parameters()).device
    half = a.items // 2
    exs = build_examples(
        tok, dev, a.items - half, a.seed, 0.55, 4000, 8000, 1, standing_frac=0.18
    ) + build_paged_examples(tok, dev, half, a.seed)
    exs.sort(key=lambda e: -(int(e.new_ids.shape[0]) + int(e.query_ids.shape[0])))
    print(f"{len(exs)} items, longest first -- the longest is the one that decides the cap")

    peaks, ooms = [], 0
    for i, ex in enumerate(exs, 1):
        torch.cuda.reset_peak_memory_stats()
        n = int(ex.new_ids.shape[0] + ex.query_ids.shape[0])
        try:
            parts = example_losses(
                model,
                loras,
                ex,
                32,
                anchor=(i % 2 == 0),
                backward=1.0 / a.accum,
                anchor_weight=1.0,
                grad_prefill=not a.no_grad_prefill,
                preserve_weight=2.0,
                grad_prefill_max_tokens=a.cap,
            )
            peak = torch.cuda.max_memory_allocated() / 2**30
            peaks.append((peak, n, ex.edit_kind, parts["grad_prefill"]))
            ooms += bool(parts["grad_prefill_oom"])
            print(
                f"  [{i:>3}] {ex.edit_kind:<14} {n:>5} tok  peak {peak:5.2f} GiB"
                f"  grad_prefill={parts['grad_prefill']}"
                f"{'  (fell back on OOM)' if parts['grad_prefill_oom'] else ''}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to survive
            print(f"  [{i:>3}] {ex.edit_kind:<14} {n:>5} tok  FAILED: {type(exc).__name__}: {exc}")
            break
        for lora in loras:
            lora.lora_a.grad = lora.lora_b.grad = None
    if peaks:
        worst = max(peaks)
        print(
            f"\nmeasured peak {worst[0]:.2f} GiB at {worst[1]} tokens ({worst[2]}); "
            f"median {statistics.median(p[0] for p in peaks):.2f} GiB; "
            f"{sum(p[3] for p in peaks)}/{len(peaks)} kept the expressive path; "
            f"{ooms} OOM fallbacks"
        )
        print(f"headroom on a 32 GiB card: {32 - worst[0]:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
