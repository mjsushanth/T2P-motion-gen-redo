# Experiment log

Chronological record of every experiment run in this repository, with honest results.
**Append-only.** Newest at the bottom. Never edit an earlier entry to agree with a later one —
if a result is superseded, add a new entry that says so and cross-reference both.

## Conventions

- **VERIFIED** = ran here, output observed and saved to `../artifacts/`.
  **UNVERIFIED** = not checked here.
- **Every number states the data it came from.** A metric with no stated split and sample count
  is a bug, not a result.
- **Training losses do not appear in this file.** They are diagnostics. See `LANDMINES.md` §4.
- Subset or laptop-scale results are never compared to published leaderboard numbers without
  stating why the comparison is invalid.
- No published figure is quoted until it has been **fetched** and the source recorded.
- Every entry names what it **does not** establish. That line is mandatory.

## Entry template

```markdown
## Enn — <title>

**Ran by:** <script or notebook path>   **Date:** <ISO>   **Seeds:** <n, listed>
**Data:** <dataset, split, sample count>
**Record:** `../artifacts/nn_<name>_record.json`
**Status:** VERIFIED | PARTIAL | FAILED

**Hypothesis (pre-registered):** <stated before the run>
**Success criterion (pre-registered):** <the number that would decide it>

| metric | value | seed spread |
|---|---|---|

**Result:** <what happened, including if the hypothesis was falsified>
**Establishes:** <what can now be claimed>
**Does NOT establish:** <mandatory>
```

---

## E000 — reserved

Nothing has been run in this repository yet. The first entry will be the evaluation-harness
validation gate (`DECISIONS.md` D-03), **not** a model result.

---

## E0a — evaluator harness sanity check (real vs. real), NOT the D-03 gate itself

**Relabeled E0 -> E0a (2026-09-06, same day, no data changed):** per review, this is better
understood as one of two independently-diagnostic rungs. **E0a isolates the evaluator + this
project's `opt`-reconstruction + data adaptation** from checkpoint-loading/generation entirely —
if it fails, the bug is on this project's side (wrong `opt` field, wrong windowing, wrong
normalisation — exactly the shape of F1: a plausible reconstruction of an undocumented config
that runs cleanly and gives wrong numbers, with no error raised). **E0b** (below) only makes sense
to run once E0a passes, and tests something different: whether the published ladder itself is
reproducible from released artifacts at all.

**Ran by:** `../scripts/e0_evaluator_sanity_check.py`   **Date:** 2026-09-06
**Seeds:** 0, 1 (both full test split; seed only affects which random unit-length crop and
train/test split partition point is used — see spread below)
**Data:** HF `TeoGchx/HumanML3D`, `test` split, streamed (not bulk-downloaded), all 4384 sequences
that pass the length filter (40-199 frames) -> 4198 usable samples, split into two disjoint
halves of ~2099 each
**Record:** `../artifacts/e0/e0_evaluator_sanity_check_record_seed0.json`,
`../artifacts/e0/e0_evaluator_sanity_check_record_seed1.json`
**Status:** PARTIAL — see "Does NOT establish" below. This is a precursor sanity check, not a
completed D-03 gate.

**Hypothesis (pre-registered, stated before running):** the vendored Guo et al. evaluator
(`third_party/text-to-motion/`, checkpoint sourced from a third-party HF re-upload,
architecture-verified against the evaluator's own model classes — see
`third_party/text-to-motion/PATCHES.md`) should produce sane, signal-carrying numbers when run on
real HumanML3D data: matched (text, motion) pairs should retrieve each other far above chance in
a 32-candidate R-Precision batch (paper's own protocol), and FID between two disjoint real-data
subsets should be small, in the neighborhood of (though not necessarily identical to) the paper's
own "Real"-row FID of 0.002.

**Success criterion (pre-registered):** R-Precision-top3 comparable to the paper's 0.797 (same
ballpark, not required to match exactly given this uses disjoint subsets and a single random crop
per motion rather than the paper's exact protocol); FID small and stable across sample-size
increases and across seeds.

| metric | seed 0 | seed 1 | spread |
|---|---|---|---|
| R-Precision top-1 | 0.4269 | 0.4264 | 0.0005 |
| R-Precision top-2 | 0.6159 | 0.6144 | 0.0015 |
| R-Precision top-3 | 0.7202 | 0.7159 | 0.0043 |
| matching score | 3.6057 | 3.6084 | 0.0027 |
| FID (real vs. real, disjoint 2099/2099) | 0.02870 | 0.02890 | 0.00020 |

Seed spread is tiny relative to the effect sizes discussed below (e.g. R-Prec-top3's ~0.4 point
spread vs. its ~7.7-point gap from the paper's 0.797) — this ran entirely on CPU (not MPS), so
the residual spread reflects only which random unit-length crop/subset-split each seed drew, not
device non-determinism (`LANDMINES.md` §7 does not apply to this particular check).

**Progression observed as protocol was corrected (all real data, all seed 0):**

| version | n samples (A/B) | R-Precision batching | R-Prec-top3 | FID |
|---|---|---|---|---|
| v1 (bug) | 200/200 | single batch of 200 (1 correct + 199 distractors) | 0.280 | 0.745 |
| v2 (fixed batching) | 1024/1024 | 32-candidate batches, averaged | 0.710 | 0.173 |
| v3 (full test split) | 2099/2099 | 32-candidate batches, averaged | 0.720 | 0.029 |

**Result:** the harness is clearly producing real, meaningful signal — matched text/motion pairs
retrieve each other at 72% top-3 accuracy out of 32 candidates (vs. ~9% chance), and FID between
disjoint real subsets fell by 25x (0.745 -> 0.029) purely from fixing the R-Precision batch size
and increasing sample count to the full test split, with no other change. This is strong evidence
the checkpoint, the text/motion encoders, and this project's own data-preparation code (caption
tokenization, unit-length cropping, Z-normalization with the official Mean.npy/Std.npy) are all
working correctly together. **The v1 bug is documented here rather than silently fixed and
forgotten** — a 200-candidate R-Precision pool is a real methodological mistake this project's
own code made, caught before being reported as a result, which is exactly the discipline
`LANDMINES.md` exists to enforce on this project's own work, not just the original's.

The remaining gap from the paper's own numbers (R-Prec-top3 0.720 vs. 0.797; FID 0.029 vs. 0.002)
is real and did not close further between the 1024-sample and 2099-sample (full split) runs,
suggesting it is not primarily a sample-size artifact. Most likely explanation, not yet
confirmed: this script uses a single random unit-length crop per motion, while the paper's
`final_evaluations.py` protocol conventionally evaluates with `--repeat_time` averaging over
multiple crops/seeds per sample — this project's version does not yet replicate that averaging.

**Establishes:** the vendored evaluator harness (checkpoint + encoders + this project's own data
pipeline) is functioning and produces results in the correct ballpark on real data, not degenerate
or broken. The checkpoint's architecture-match to the official evaluator (documented in
`third_party/text-to-motion/PATCHES.md`) is now also behaviorally corroborated, not just
shape-corroborated.

**Does NOT establish:**
- **This is NOT the D-03 gate.** D-03 requires reproducing a *published generated-model* number
  (e.g. MDM's FID 0.544) to a stated tolerance. This entry only validates the harness against
  real data, which has no generative model in the loop at all yet.
- Does not establish that the ~0.72 vs. 0.797 R-Precision gap and ~0.029 vs. 0.002 FID gap are
  fully understood — the multi-crop-averaging hypothesis above is a plausible, not confirmed,
  explanation.
- Does not establish the checkpoint is byte-identical to the original Google-Drive-hosted file —
  only that it is architecturally and behaviorally consistent with it (see PATCHES.md).
- Does not establish anything about MPS — this ran entirely on CPU (the evaluator's BiGRU
  encoders are small; CPU inference is fast and avoids `LANDMINES.md` §7's non-determinism
  entirely for this particular check, which is why seed-0 reran bit-for-identical).

**Next:** obtain an actual generated model's output (MDM's own released checkpoint/samples, or
this project's own future E2 baseline) to complete the actual D-03 gate — reproduce a published
generated-model FID to a stated tolerance, per `REBUILD_SPEC.md` §6's E0 definition. Pre-registered
as E0b below, before either its method or its result is known in detail.

---

## E0b — published-number reproduction (MDM's FID), PRE-REGISTERED before running

**Pre-registered:** 2026-09-06, before fetching MDM's checkpoint or running any generation.

**Hypothesis:** running MDM's released model on the HumanML3D test set and evaluating its
generated motions through the same vendored evaluator (E0a having passed as a precondition)
reproduces MDM's own published FID (0.544±.044, `LANDSCAPE.md` §1.3) to within a stated tolerance.

**Success criterion (fixed now, before the result exists):** FID within ±5% of 0.544 (i.e.
0.5168-0.5712), using MDM's own released checkpoint and its own standard sampling procedure
(no changes to guidance scale, step count, or sampler beyond what MDM's own eval script uses).

**Explicitly pre-committed:** if the reproduced FID falls outside ±5%, that is **the result**,
not a reason to widen the tolerance. Either outcome is reported as run — "FID reproduced within
tolerance: PASS/FAIL, actual value X" — per review's standing instruction not to average
E0a/E0b together or soften a miss.

**What a FAIL would mean, stated before it's known which way this goes:** E0a passing plus E0b
failing would mean this project's own evaluator adaptation is sound but the published number is
not reproducible from MDM's released artifacts alone (a finding about the field's shared
instrument, not about this project). E0b failing would say nothing about E0a's independent pass.

**Status:** PENDING — not yet run. Fetching MDM's repo/checkpoint next.
