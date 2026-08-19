# Marathon — Phase 3: stitched-KV consistency fine-tuning

**Status:** Design + pilot · **Date:** 2026-08-19 · **Companion:** [DESIGN.md](../DESIGN.md) (doc 0001), [PLAN.md](PLAN.md), [phase1-report.md](phase1-report.md), [findings.md](findings.md)

## Why the original Phase 3 is mostly already done

DESIGN.md framed the research bet as *the trust contract*: the model is handed a diff plus new input, told that anything absent from the diff is unchanged by definition, and trained to stop re-reading the baseline. The stated failure without training was that "a model given only a diff will hallucinate or re-request the baseline."

Phase 1 obtained that outcome **without training, and more completely than the text-level version could have.** Position-shifted KV reuse does not hand the model a diff and ask it to trust the absence; it hands the model a *cache*, and the model's attention runs over the full history exactly as if it had re-read it. The baseline is never re-tokenized, never re-prefilled, and never re-transmitted, so the cost goal is met — but the model also never has to be persuaded of anything, because nothing about its input distribution changed. On the 144-session eval, re-rotated reuse forwards 1.5–1.6% of tokens for median KL 0.003 against full recompute, with planted-fact retrieval at 105/111 versus full recompute's 106/111. A text-level delta format would have been strictly worse on both axes: it changes the model's input distribution (needing training to recover), and it loses information the KV path keeps for free.

So the specific bet "fine-tune the model to consume delta-formatted TEXT" is superseded. What is *not* superseded is the harder, more general form of the same idea, and Phase 1 handed us a sharp instance of it.

## What actually remains

**(a) The residual failure class.** Position-shifted reuse has exactly one failure mode across 144 sessions, 469 items, 5→7 edit kinds and 7 query types: an edit to a **governing** span — the system prompt, a standing instruction, a persona or output-format directive. There, mean KL against full recompute is ~9× the non-governing level (0.0264–0.0350 vs 0.0020–0.0167), and it is the *only* class that produces items above KL 0.2. The 2×2 that broke the position/flag confound settles the cause: holding position and |S| fixed and flipping only the governing flag moves mean KL 9×; holding the flag fixed and moving the edit from the front to the middle moves it by nothing. spearman(KL, |S|) = −0.028; spearman(KL, |δ|) = +0.237.

The mechanism is not mysterious. Re-rotation fixes *where* the reused suffix `S` sits; it cannot fix *what `S` attended to*. Every token in `S` was computed while the old instruction was in context, and its residual stream still carries the model's response to it. When the instruction is replaced, the suffix keeps whispering the old one. Selective recompute is the obvious repair and it does not work: the hand-built `dep-instruction` scenario improves first-token KL 0.3492 → 0.0384 at `first-32` and → 0.0177 at `first-512` (10% of `S`, i.e. most of the win given away), and even then agreement with the reference is restored in only half the cases. `blend-r0.30` is *worse* than reuse-all (kl_first 0.1761 at 35% effective recompute) because its layer-1 K-deviation selector barely registers an instruction flip. The contamination is diffuse across the whole suffix, not concentrated in a prefix of it, so there is no cheap subset to recompute.

The current production answer is `reuse_plan`'s `repair` policy: flag governing edits and recompute. That is correct and it is what ships. It is also a capitulation exactly where sessions are longest and most valuable — a system prompt or standing-instruction edit is the one edit that invalidates the entire history.

**(b) The general form: robustness to stitched caches.** Governing edits are the sharpest case of a class Phase 1 keeps running into, in which reused KV is *placed* correctly but *conditioned* on something that is no longer true:

- **Relocated blocks.** Transplanting KV for a block that moved by δ = −10,153 produced garbage (`'1'`, `'2'`, `' content used to grow the context…'` instead of the planted code), and `--repair-first 256` did not fix it. Relocations are therefore recomputed, at 4.0× instead of the transplant's speed. The boundary between the δ = +186 that works and the δ = −10,153 that does not is uncalibrated.
- **Large δ.** spearman(KL, |δ|) = +0.237 overall and +0.407 among non-governing items: |δ| is the strongest single continuous predictor in the eval. Nothing in the current design bounds it.
- **Hybrid washout.** On Qwen3.5-4B's Gated-DeltaNet layers an edit far from the end of a long context is washed out of the decaying recurrent state, and reusing the end-of-context state alone is 5–25× worse in KL. The fix is recomputing the first ~256 tokens of the suffix, with M uncalibrated.

All three are the same shape: *the reused state is a good approximation of a state that was computed under a slightly different history, and we would like the model to not care.* Today the answer is always "recompute more". Phase 3's bet is that a small adapter can buy some of that back, letting reuse be applied more aggressively than a purely conservative planner ever could.

## Method: stitched-KV consistency fine-tuning

Self-distillation, where the teacher is the model's own clean-context behaviour and the student is the same model reading the cache the serving path actually builds.

```
teacher   frozen base weights, full recompute of the EDITED context   -> reference distribution
student   base weights + LoRA, forward run against the STITCHED cache -> P verbatim, E' fresh,
                                                                        S re-rotated by delta
loss      KL(teacher || student) over N continuation tokens, teacher-forced on the teacher's
          own greedy continuation                        [ this is the eval's `klmean` ]
        + w * KL(teacher || student-on-a-CLEAN-cache)    [ anchor: 0 at init, by construction ]
```

Three properties make this the honest version of the experiment rather than a proxy:

**The training objective is the evaluation metric.** `kvshift_eval`'s headline number is mean KL over 32 teacher-forced continuation tokens against a full recompute. That is literally the loss. There is no gap between "the loss went down" and "the reported metric went down", and no reward model, preference data, or synthetic "correct answer" to argue about — the target is what the model itself does when it is allowed to read the whole context.

**The anchor term measures the damage in the same units, with no floor.** Running the student on a clean, unstitched cache and taking the same KL against the teacher is exactly zero for the base model, because the base model *is* the teacher. Any nonzero value is adapter-induced drift on ordinary contexts, in the same units as the thing being bought. If clean-KL rises as fast as stitched-KL falls, the method has bought nothing and the pilot says so.

This only works if the zero is a real zero, and the first version of it was not: prefilling the anchor separately from the teacher put a ~0.005 bf16 floor under the metric — larger than the damage it has to resolve — and the first pilot launch reported `clean_kl = 0.0049` at step 0, where an identity adapter must give 0. Teacher and anchor now share one function (`_clean_sequence`), so at identity they are bit-identical and the reading is 0.0000. This also means the anchor does *not* inherit the ~0.0015 numerical floor the eval's `prefix-equiv` row sits at, which is the price of comparing a step-by-step greedy decode against a batched teacher-forced one; a test pins the exact-zero property, because a refactor that gives the two paths separate-but-equivalent implementations would reintroduce the floor silently. The anchor's prefill runs under `no_grad` — its gradient comes from the continuation positions only, which is also what keeps an 8B anchor affordable.

**It is differentiable end to end, and cheaply.** RoPE re-rotation is a multiplication by a constant `cos/sin` pair; segment placement is a copy. Neither has parameters and both pass gradients. The reused KV *itself* is a constant — it came from a previous turn and nothing can backprop into it, which is what makes the whole thing cheap: no backward pass over the 5k-token history. Gradients reach the adapter through two live paths only: the query and continuation tokens' forward (how the model *reads* a stale suffix — q/o proj) and the freshly computed span `E'` written into the cache inside the graph (what the model *writes* so the stale suffix matters less — k/v proj). `stitch_train.GradScatterLayer` exists solely because `kvshift.ShiftCache`'s in-place scatter is correct under `no_grad` and opaque to autograd; the two are asserted to place identical tensors in `tests/test_stitch_train.py`.

**The do-no-harm term (added after the 8B run).** The 8B result cut governing KL 43% while *raising* non-governing KL 37%, even though non-governing items were half the training mix. They were present but defenceless: a non-governing item starts at `klmean` ~0.005 against a governing item's ~0.037, so plain `stitch_kl` hands the already-fine cases almost no gradient and the optimiser spends them. The fix changes what is asked of them rather than how many there are. A governing item is asked to get better; a non-governing item is asked only not to get worse than the frozen base already is on that same item:

```
governing      loss = klmean(teacher || student_stitched)
non-governing  loss = w * relu( klmean_student - klmean_base - slack )
```

zero while the adapter is at least as good as the base, growing only on regression. It costs one extra base-model stitched forward per non-governing item (adapters off, no grad). A hinge rather than a KL-to-base term on purpose: pinning non-governing items *to* the base would also forbid the incidental improvements, and what is wanted is a floor, not a leash.

**Mid-training checkpoints.** The 8B run left an ambiguity it could not resolve — split-half training medians moved the wrong way while held-out KL improved 43%, and with no intermediate measurement there was no way to distinguish a harder second half from an improvement that was not really coming from training. `--checkpoint-every` now saves the adapter and runs a held-out mid-training eval (its own seed, disjoint from both train and final eval), so the curve is measured rather than inferred.

**Memory and the two halves of the method.** Gradients can reach the fresh span `E'` (so `k_proj`/`v_proj` learn what to *write* into the cache) or stop at the continuation (so only `q_proj`/`o_proj` learn how to *read* a stale suffix). The first is strictly more expressive and costs a retained full-length K and V per layer — ~2.4 GB at 8B/8k, which is what exhausted the GPU 81 items into the first 8B attempt. It is `--grad-prefill`, and it is bounded by `--grad-prefill-max-tokens` (default 6000) rather than by gradient checkpointing: checkpointing re-runs each decoder layer during backward, and those re-runs would re-enter the cache scatter this module performs. Items over the cap fall back to the cheap path and are counted, so a run always reports how much of the expressive half it actually got. **The 8B numbers so far come from the cheap half only.**

**Data.** `kvshift_eval.build_item` already builds the population the eval is measured on: seeded multi-turn sessions over three families (repo code, repo prose, generated fact tables), three codes planted before/inside/after the edit, one staged single-span edit of a chosen kind, and a query pool including the fact questions and an `obey` question that only a governing edit can break. `stitch_train.build_examples` calls it with a `gov_frac` split — half governing (`governing` + `mid-governing`), half drawn from `fact`/`early-fact`/`rewrite`/`insert`/`delete` as regularisation, so the adapter cannot buy governing robustness by wrecking the four kinds that already work at KL ≈ 0.002. **Train and eval seeds are different and the eval seed is never trained on**; a test asserts the two populations differ.

**LoRA, hand-rolled.** `peft` is not in the venv and this does not justify adding it: `LoRALinear` is a wrapped `nn.Linear` with a zero-initialised `B` (identity at init) and an `enabled` flag. The flag is load-bearing beyond tidiness — teacher and student are the same weights in the same allocation, so an 8B teacher costs nothing extra, and "adapters off" is bit-for-bit the base model, which is what lets one eval run report the before and after columns on identical examples.

## Exit criteria

**Pre-registered metric.** Every KL number in every criterion below is **`klmean`**: the mean KL over 32 teacher-forced continuation tokens of the *reference's own greedy output*, student against the frozen base model's full recompute of the edited context, under `reuse-all`. Not `kl_first`, not free-running exact match. This is stated once here because the first round of criteria did not state it and it cost a real comparison: criterion 4 quoted 0.3492, which is a **`kl_first`** number from the 2026-08-18 probe, and the 8B run reported `klmean`, so "did it move?" had no well-posed answer. `kl_first` and `klmean` differ by more than an order of magnitude on the same item — `dep-instruction` reuse-all was `kl_first` 0.3492 against `klmean` 0.0118 — so they are never to be compared across criteria.

**Reported statistics.** Wherever a bucket is summarised, report **mean and median together**, plus the counts over 0.05 and over 0.2. The two disagree in a way that matters: on the same base model the governing/non-governing ratio was 8.71× mean / 4.03× median at n=48 and 7.25× / 2.27× at n=60. The mean tracks the tail (which is what forces `reuse_plan` to refuse) and the median tracks the typical item (which is what a user experiences), and a fix that moves only one of them is only half a fix.

**Sample size.** Ratios at n≈50 are unstable enough that the base model alone moved the median ratio from 4.03× to 2.27× between two samples. Gate runs use **≥120 held-out items** and report the ratio with its n; anything smaller is a smoke test, not a gate.

Measured on **held-out seeds** never trained on:

1. **The failure class closes.** Governing/non-governing `klmean` ratio ≤ **2× on the mean and ≤ 2× on the median**, from a base measured on the *same* population in the same run (adapters off), and no held-out governing item above `klmean` 0.2. Report both ratios and both n. *(8B, n=60: 7.25× → 3.01× mean, 2.27× → 1.42× median, one item at 0.278 — fail.)*
2. **Clean context is undamaged.** Clean-context `klmean` of the adapted model against the base ≤ **0.002 on the mean**, with planted-fact accuracy equal to the base within one item. 0.002 is a policy choice, not a measurement floor — the anchor reads exactly 0 at identity — set at the level `prefix-equiv` sits at against full recompute (0.0014–0.0017). *(8B: 0.0024 mean / 0.0015 median — marginal fail.)*
3. **No regression on what already works.** Non-governing `klmean` no worse than the base model's by more than **20%** (the seed-1234-vs-1235 spread, 0.0142 vs 0.0119), on mean *and* median, with items over 0.05 not increased. *(8B: +37% mean, +46% median — fail.)*
4. **The dependent-instruction scenario moves.** `klmean` on `kvshift_probe`'s `dep-instruction` questions falls by ≥ 30% against the same base, with the base measured in the same run. The old formulation ("toward the `first-512` level of 0.0177") is retired: that is a `kl_first` target and cannot be checked against a `klmean` measurement. *(8B: 0.0133 → 0.0093 and 0.0178 → 0.0104 — pass.)*
5. **The win is kept.** No change to tokens forwarded — the adapter changes weights, not the reuse plan — so any KL improvement is free at serving time apart from the LoRA's own FLOPs. *(Holds by construction.)*

A result that fails (2) or (3) is a failure regardless of (1): *efficiency that changes answers is a regression, not a win*, and that rule is not suspended because the mechanism is now training instead of caching. The 8B run is exactly that case — a real 43% mean / 61% p95 cut on governing edits, disqualified by a 37% regression everywhere else.

If the criteria are met, the payoff is concrete: `reuse_plan` can downgrade governing edits from `repair` to `reuse`, which is the difference between 1.5% and 68% of tokens forwarded on precisely the edit that invalidates the most history.

## Risks, stated before the run

- **Overfitting to synthetic edits.** The training population comes from five hand-written instruction pairs (French→German, lowercase→CAPITALS, `<<END>>`→`<<STOP>>`, …) and repo text. An adapter that learns "when the system prompt mentions German, ignore the suffix" generalises to nothing. Held-out seeds re-draw the *sessions* but not the instruction pool, so criterion (1) is weaker than it looks; a genuinely held-out instruction set is the first thing to add if the pilot is positive.
- **Damaging clean-context quality.** The anchor term is the mitigation and criterion (2) is the gate. This is the most likely way the method fails honestly.
- **The teacher is not ground truth.** It is the model's own full-recompute behaviour, which is sometimes wrong — the dependent-edit study found the 8B reference *ignoring* the German instruction while the reuse paths obeyed it. Distilling toward full recompute therefore distills toward the model's existing instruction-following, warts included. That is the right target for this phase (the claim under test is "reuse changes no answers", not "the model is good"), but it means a KL improvement is not an accuracy improvement.
- **KL is a soft target and exact-match has a ~20% instability floor.** The eval's own calibration shows `prefix-equiv` — which is KL ≈ 0 by construction — reaching only 0.80–0.86 exact match. Nothing here should be argued from exact match.
- **The pilot's scale is not the deployment's scale.** 0.6B first for iteration, 8B if time; the serving model is 14B-FP8, and an adapter is model-specific. A positive pilot is evidence the mechanism exists, not a shippable artifact.

## What the pilot was, and what it showed

The smallest honest experiment: ~600 sessions at 3–6k tokens, half governing, one query each, 32 continuation tokens, LoRA r=16 on q/k/v/o, one epoch, AdamW at 1e-4 with the anchor at weight 1.0 every other step; evaluated on 120 held-out sessions from a different seed.

**Run 2026-08-19 on Qwen3-0.6B. The result is negative, and the negative is about the testbed.** On the held-out population a governing edit cost mean KL 0.0034 against non-governing's 0.0023 — a 1.50× ratio where the 8B eval measured ~9×, with zero of 120 items over KL 0.05 and `dep-instruction` scoring 0.0038/0.0078 instead of breaking. The failure class was simply not present, so the adapter had no signal to exploit: stitched KL did not improve (0.0028 → 0.0033) and clean-context drift cost 0.0031, failing criteria 2 and 3 and leaving criterion 1 untestable. Three causes at once — the 0.6B model follows standing instructions weakly enough that a stale copy of one barely perturbs it; sessions were 3–5k rather than 4–8k; and `build_examples` took only `queries[:1]`, which `kvshift_eval` always fills with `fact-at`, so the `obey` query never ran. The full table and verdict are the 2026-08-19 entry in [findings.md](findings.md).

What the run *does* establish is that the instrument is sound and worth pointing at the real population: the differentiable stitch places tensors identical to the serving path's, gradients reach the adapter through a stitched cache, the split backward matches the summed one, and the anchor reads exactly 0 at identity.

**The 8B run (2026-08-19).** With the query pool fixed and the regime corrected, the base model shows the failure at 8.71x (48 items) and the adapter then cuts governing-edit KL 43% on the mean and 61% at p95 on 60 held-out sessions, halving the items over 0.05 and over 0.2. It fails criteria 1, 2 and 3: the mean ratio lands at 3.01x rather than <=2x, clean drift is 0.0024 against a 0.002 budget, and non-governing edits regress 37%. Promising, not shippable. Two lessons for the criteria themselves - criterion 4 never said whether it meant `kl_first` or `klmean` (the 0.3492 it quotes is `kl_first`, and this run reports `klmean`), and the median-ratio target is too noisy to gate on at n=60, since the same base model gives 4.03x on 48 items and 2.27x on 60. Full table and next steps in [findings.md](findings.md).

@acrosley 2026-08-19
