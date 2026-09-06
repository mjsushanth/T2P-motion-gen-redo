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

**CORRECTION (2026-09-06, per review SUP-20260906-21 — this paragraph's explanation was wrong,
kept above rather than deleted, per the append-only convention):** E0b's own ground-truth numbers
(same evaluator, same checkpoint, same dataset, but using **MDM's own upstream
`Text2MotionDatasetV2` loader** instead of this entry's hand-built `prepare_sample`/`opt`
reconstruction) landed at R-Prec-top3 0.7969 and matching score ~2.90-2.98 — both close to the
author's own reference (0.7977, 2.9758), not close to this entry's 0.720/3.6057. **Since the only
thing that differs between this entry and E0b's ground-truth path is which data-loading code
built the batches, the multi-crop-averaging hypothesis above was not the cause of this entry's
own R-Precision gap — this entry's own hand-rolled data pipeline had a real, uncorrected bug.**
This is precisely the F1-shaped risk that was flagged when the `opt`-reconstruction work for this
script began (a plausible reconstruction that runs cleanly and produces wrong numbers, with no
error raised) — it did bite here, just not fatally, and the move to MDM's upstream loader for
E0b is what exposed it. The specific bug in this script's own crop/normalisation/windowing logic
has not been further isolated as of this correction; recorded as an open item, not resolved.

**Establishes:** the vendored evaluator harness (checkpoint + encoders + this project's own data
pipeline) is functioning and produces results in the correct ballpark on real data, not degenerate
or broken. The checkpoint's architecture-match to the official evaluator (documented in
`third_party/text-to-motion/PATCHES.md`) is now also behaviorally corroborated, not just
shape-corroborated.

**Does NOT establish:**
- **This is NOT the D-03 gate.** D-03 requires reproducing a *published generated-model* number
  (e.g. MDM's FID 0.544) to a stated tolerance. This entry only validates the harness against
  real data, which has no generative model in the loop at all yet.
- Does not establish that the ~0.72 vs. 0.797 R-Precision gap is fully understood — the
  multi-crop-averaging explanation above is now known to be wrong (see correction above); the
  gap traces to this script's own hand-built data pipeline, not to missing crop-averaging, and
  the specific bug has not yet been isolated. The ~0.029 vs. 0.002 FID gap is separately and
  adequately explained by small-n covariance bias (`LANDMINES.md` §14), unaffected by this
  correction.
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

**Status:** RUN, 2026-09-06. Result appended below — the hypothesis/criterion above is unedited
from its pre-registered form.

**Ran by:** `../scripts/e0b_mdm_reproduction.py`, MDM's own `humanml_trans_enc_512/model000475000.pt`
checkpoint ("best model used in the paper," per the official README), MDM's own
`evaluation_parser()`/`create_model_and_diffusion()`/`evaluation()` (this project's driver script
reuses these directly rather than reimplementing them, so every hyperparameter — `diffusion_steps:
1000`, `arch: trans_enc`, `guidance_param: 2.5`, `cond_mask_prob: 0.1`, `latent_dim: 512` — loads
from the checkpoint's own bundled `args.json`, not a hand-reconstructed config).
**Data:** 128 materialized HumanML3D test-split samples (streamed from HF `TeoGchx/HumanML3D`,
not bulk-downloaded), REDUCED from the paper's ~1000-sample protocol — the checkpoint's own
bundled evaluation log states the full 20-replication protocol took "about 12 Hrs" on the
author's hardware; infeasible to match at full scale on this machine (CPU only, no dedicated
GPU) — this run alone took ~39 minutes of CPU-bound generation for 4 batches of 32.
**Record:** `../artifacts/e0/e0b_mdm_reproduction_record.json`
**Seeds:** 1 (seed=10, MDM's own default — not varied; a single, very long run, not a
multi-seed comparison; seed spread is therefore UNKNOWN for this result, stated as a gap, not
glossed over)

| metric | ground truth (this run) | vald / generated (this run) | paper's "Real" row | paper's MDM row |
|---|---|---|---|---|
| R-Precision top-3 | 0.7969 | 0.7578 | 0.797±.002 | 0.611±.007 |
| FID | **0.1339** | **1.0731** | 0.002±.000 | **0.544±.044** |

**Result:** measured `vald` FID = **1.0731**. Pre-registered success criterion was FID within
±5% of 0.544 (0.5168-0.5712). **1.0731 is outside that band — FAIL**, reported as such, not
softened, per the pre-registered commitment.

**A real caveat that applies symmetrically, not an excuse for the FAIL:** this run's own
ground-truth-vs-itself FID (0.1339) is also far above the paper's ~0.002 — direct, self-contained
evidence that 128 samples is too few for a numerically stable FID estimate in this evaluator's
512-dimensional embedding space (FID's covariance estimate needs substantially more samples than
the embedding dimension to avoid inflation; `E0a`'s own progression table showed the identical
effect: FID dropped from 0.745 at n=200 to 0.029 at the full n=2099 test split, purely from
sample-size). **This means the raw 1.0731 number should not be read as "MDM is 2x worse than
claimed"** — it is at least partly an artifact of evaluating at n=128 instead of ~1000. What it
*does* still support: even accounting for that inflation, the *relative* gap between this run's
ground-truth FID (0.1339) and vald FID (1.0731) — roughly 8x — is in the same rough direction as
the paper's own gap (0.002 to 0.544, roughly 270x, though these ratios are not on a scale where
direct comparison is sound either). The honest summary is: **this specific reduced-scale run
cannot distinguish "the pipeline correctly reproduces MDM" from "128 samples is too few to tell,"**
and resolving that needs either a much larger sample count (which needs much more wall-clock time
than was spent here) or a properly-sized ground-truth reference FID computed once, held fixed,
and compared against — the second option is cheaper and is the recommended next step if this
gate is revisited.

**A process failure, disclosed rather than hidden:** the run's `Diversity` metric computation
crashed on an assertion (`diversity_times=128` exactly equals `num_samples_limit=128`, failing
`metrics.py`'s `activation.shape[0] > diversity_times` check) — an off-by-one in this project's
own driver script, not in MDM's or the evaluator's code. FID and R-Precision had already been
computed and printed before the crash, so they are not affected, but this result was assembled
by hand from the run's stdout after the crash prevented the script's own JSON-writing code from
executing — stated explicitly since it's a lower-confidence provenance path than a clean
programmatic write, even though the numbers themselves were read directly off the log, not
estimated or recalled.

**Establishes:** MDM's checkpoint, code, and this project's vendored evaluator all run together
end to end without further blocking errors (three real macOS/dependency issues were found and
patched along the way — gated SMPL body model, `spawn` vs `fork` multiprocessing, and this
project's own caption-parsing bug in `materialize_humanml3d_test_subset.py` — all documented in
`third_party/motion-diffusion-model/PATCHES.md` and this project's `LEDGER.md`). At n=128, single
replication, the measured FID does not fall within the pre-registered tolerance of the published
number.

**Does NOT establish:**
- That MDM's published 0.544 is unreproducible in general — only that *this specific
  reduced-scale attempt* did not land in tolerance. The sample-size caveat above means this run
  cannot cleanly separate "real discrepancy" from "n=128 is too few," and the honest verdict is
  FAIL-as-measured, not "MDM's number is wrong."
- Anything about seed spread (only one replication was run — `LANDMINES.md` §7's seed-spread
  discipline could not be applied here given the ~39-minutes-per-replication cost).
- That this project's own future model (once one exists, per D-18/D-20's task reframing) would
  show the same gap — E0b tests only whether *MDM's own released checkpoint* reproduces its own
  published number under this project's harness, not anything about a model this project builds.

**CORRECTION (2026-09-06, per review SUP-20260906-15, P0):** this entry originally said here
"D-03's gate is satisfied by E0a per the director's Stage 2 review." **That was wrong** — the
Stage 2 review's gate condition named the *reproduction* (this entry, E0b) as the actual gate;
E0a's own title already said "NOT the D-03 gate itself," and that was correct then and still is.
D-03 itself has a documented fallback for exactly this situation: *"reproduce one published
HumanML3D figure to a stated tolerance, or explicitly downgrade every number to
internally-comparable-only."* The reproduction (above) missed its tolerance. **D-03 is therefore
UNRESOLVED** — not passed (the reproduction failed its stated tolerance), and not definitively
failed either (n=128 cannot cleanly settle whether the miss is a real discrepancy, per SUP-16/17
below). Per D-03's own fallback clause: **every number from E1 onward is internally-comparable-
only until this resolves**, and that must be stated loudly wherever such numbers are reported
(`RESULTS.md` when it exists, and any experiment-log entry in the meantime).

**Diagnostic update (2026-09-06, per review SUP-20260906-16/17):** the "n=128 is too few" framing
above is not sufficient on its own. Subtracting this run's own measured ground-truth FID bias
(0.1339 - 0.029 [E0a's full-scale ground-truth FID] = 0.105) from the generated FID
(1.0731 - 0.105 ~= 0.968) still leaves the result ~1.7x outside the tolerance band — sample-size
bias alone does not rescue the FAIL (bias is not strictly additive across distributions, so this
is indicative, not decisive). More pointed: this run's ground-truth R-Precision-top3 (0.7969)
matched the published value (0.797) to three decimals, showing n=128 is *not* too few to measure
R-Precision reliably and independently confirming this project's evaluator/data pipeline are
correct. But this run's *generated* R-Precision-top3 (0.7578) is 0.147 points *better* than the
paper's own reported generated value (0.611), while this run's generated FID (1.0731) is roughly
2x *worse* than the paper's (0.544) — better text-alignment paired with worse distributional
realism is the textbook signature of sampling under a different effective classifier-free
guidance strength than the reference used, and was flagged as the most likely lead to check
before spending more compute.

**That check was run, cheaply, before any further generation (per SUP-18):** empirically
instantiated the model/diffusion (no sampling) and printed the actual runtime values rather than
trusting code-reading alone. Confirmed exactly matching the reference protocol: `guidance_param
= 2.5` (matches the checkpoint's own bundled log filename, `..._gscale2.5_...`, byte-for-byte);
`diffusion.num_timesteps = 1000` and `len(use_timesteps) = 1000` (no DDIM/respacing shortcut —
`create_gaussian_diffusion` hardcodes `timestep_respacing = ''`, so the full step count is always
used); sampler is `p_sample_loop`, not `ddim_sample_loop` (`use_ddim = False` is hardcoded in
`comp_v6_model_dataset.py`, matching the reference); and the model is correctly wrapped in
`ClassifierFreeSampleModel`. **SUP-17's specific hypothesis (wrong effective guidance/step count)
does not hold — all three checked values match the reference exactly.** This rules out the
cheapest, most likely driver-bug explanation; the remaining candidates are single-replication
stochastic variance (only 1 replication was run — the paper's own 20 replications exist partly
to average out exactly this kind of run-to-run generation noise, which this entry's single run
cannot distinguish from a real effect) and the still-open checkpoint-provenance confound
(SUP-12 — architecture-verified, not cryptographically verified against the original).

**A cheaper diagnostic than any rerun, per review SUP-20260906-15..19: the checkpoint's own
bundled 2022 evaluation log already reports R-Precision, not just FID.** Re-read it directly
rather than rerunning anything:
```
========== R_precision Summary ==========
---> [vald](top 1) Mean: 0.3195 CInt: 0.0048;(top 2) Mean: 0.4978 CInt: 0.0043;(top 3) Mean: 0.6110 CInt: 0.0067;
```
**`R-Prec-top3 = 0.6110`, matching the paper's published 0.611 to three decimal places.** This
resolves the ambiguity this entry originally left open: the published 0.611 is *not* an outlier
in MDM's own artifacts and this checkpoint *does* reproduce it under the correct (full-scale)
protocol — so the earlier framing here ("either the paper's number is odd, or mine is") is
settled in favor of **mine is**. Combined with the guidance/step/sampler check above (all
clean), the discrepancy is not explained by any driver misconfiguration found so far. The
leading remaining hypothesis, not yet confirmed: this run's 128-sample subset was **not** a
random draw comparable to the reference protocol's ~1000-sample pool over the full ~4384-sequence
test set — it was the first 128 rows (in HF streaming order) that passed the length filter, and
R-Precision's batch-wise retrieval difficulty depends on the *composition* of the 32-candidate
pool a caption is matched against, not just on n (this is a distinct mechanism from §14's FID
covariance-bias finding, which *is* purely an n effect) — a smaller, non-randomly-ordered subset
could plausibly present an easier or harder retrieval task than the reference's larger, properly
shuffled pool, in either direction. This is stated as the leading hypothesis, not a confirmed
finding — distinguishing it from single-replication generation variance would need either a
proper random draw from the full test set at comparable scale, or a second independent
replication at the same n=128 to see whether the same pattern recurs.

**Round 2 (2026-09-06), per review SUP-20260906-20..24: the bundled log settles more than round 1
knew.** The checkpoint's bundled 2022 log is 20 full-scale replications of *this exact
checkpoint*, and it matches **every** published number, not just FID: GT R-Prec-top3
0.7977±.0022 (paper 0.797±.002), vald R-Prec-top3 0.6110±.0067 (paper 0.611±.007), GT FID
0.0016 (paper ~0.002), vald FID 0.5443±.0442 (paper 0.544±.044), both matching scores. **MDM's
published 0.611 is confirmed correct, not an outlier — the earlier framing in this entry that
entertained "maybe the paper's number is the odd one" is retracted; the discrepancy is entirely
on this project's side.** The 20-replication spread for vald FID (individual replications
0.5323-0.7114) also directly answers whether single-replication noise could explain a miss this
large: **it cannot** — our 1.0731 sits far outside that entire spread, and a second replication
would not have told us anything the author's own 20 already didn't.

**This also relocates a claim in `E0a`'s own entry.** Comparing E0a's ground-truth numbers
(hand-built data pipeline: R-Prec-top3 0.720, matching score 3.606) against this entry's
ground-truth numbers (MDM's own upstream `Text2MotionDatasetV2` loader: R-Prec-top3 0.7969,
matching score 2.8995-2.9758 across two runs here, both near the author's 2.9758/0.7977) shows
**E0a's earlier "multi-crop-averaging" explanation for its own R-Precision gap was wrong** — same
evaluator, same checkpoint, same dataset, and the only thing that differs between E0a and this
entry is which data-loading code built the batches. **E0a's hand-rolled `opt`/data pipeline had a
real, uncorrected bug** — exactly the F1-shaped risk flagged when that reconstruction began.
E0a's own entry is corrected separately with a pointer to this finding, per the append-only
convention (not silently rewritten).

**With the ground-truth path now confirmed correct here, the defect is isolated to generation.**
Checked the three cheapest leads for the "generated R-Precision better than published, generated
FID worse" signature, per review SUP-20260906-21, by modifying the driver
(`scripts/e0b_mdm_reproduction.py`) to **cache the generated motions and their metadata this
time** (per SUP-23 — the expensive intermediate, not just the cheap final metric) and rerunning
once (n=128, 1 replication, same ~39-minute cost as before):

1. **Generated motion length distribution vs. ground truth** — nearly identical (generated:
   min 44, max 196, mean 136.66; ground truth: min 44, max 196, mean 137.34 — means within 0.7
   frames of each other; 28/128 generated at the max length, consistent with natural variation
   rather than a fixed/max-length generation bug). **Ruled out.**
2. **Unique caption count** — 128/128 captions in this subset are unique. No duplication.
   **Ruled out.**
3. **Caption-to-motion pairing at scoring time** — confirmed by direct code inspection that
   `CompMDMGeneratedDataset.__getitem__` reuses the exact same caption/tokens dict entry that was
   attached to each motion at generation time (`self.generated_motion[item]`); no
   re-fetch-by-index step that could desynchronize conditioning text from scored text.
   **Ruled out.**

This second run's own numbers (GT R-Prec-top3 0.8125, vald R-Prec-top3 0.7578 — top-3 identical
to round 1's 0.7578; GT FID 0.1428, vald FID 1.3997) **reproduce the same qualitative pattern as
round 1** despite a different random draw (the added diagnostic code shifted the RNG state before
`gt_loader`'s own `shuffle=True`, so batch composition differs slightly between the two runs) —
further evidence this is not single-run noise. Full diagnostic and both runs' numbers in
`../artifacts/e0/e0b_mdm_reproduction_record_v2.json` and the cached generation itself in
`../artifacts/e0/e0b_generated_cache/` (available for any future re-analysis at zero
regeneration cost).

**Next:** D-03 remains UNRESOLVED; downstream numbers are internally-comparable-only until it
resolves. All of round 2's cheap leads are now exhausted without finding a fixable driver bug.
The one hypothesis not yet directly tested: this project's 128-sample subset was selected as the
first rows in HF streaming order that passed the length filter, not a random draw comparable to
the reference protocol's much larger pool drawn from the full ~4384-sequence test set — testing
this would need either a properly randomized subset draw at the same n, or accepting the full
n~1000 scale (~5 CPU-hours) to remove the question entirely. Holding before spending either,
to report the exhausted-leads status first. Still open, lower priority: fix the
`diversity_times` off-by-one (now free to do without regeneration, since motions are cached).
