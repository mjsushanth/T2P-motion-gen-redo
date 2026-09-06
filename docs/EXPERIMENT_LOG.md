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
to round 1's 0.7578; GT FID 0.1428, vald FID 1.3997) reproduce the same qualitative pattern as
round 1, despite ground truth moving (the added diagnostic code shifted the RNG state before
`gt_loader`'s own `shuffle=True`, so ground-truth batch composition differs between the two runs).
Full diagnostic and both runs' numbers in `../artifacts/e0/e0b_mdm_reproduction_record_v2.json`
and the cached generation itself in `../artifacts/e0/e0b_generated_cache/` (available for any
future re-analysis at zero regeneration cost).

**CORRECTION (2026-09-06, per review SUP-20260906-25): "further evidence this is not single-run
noise" above was wrong and is retracted.** `fixseed(args.seed)` with MDM's default `seed=10`
makes generation itself **deterministic**. Round 1 and round 2's vald R-Prec-top3 were identical
to four significant figures (0.7578 both times, 97/128 both times) precisely *because* they are
the same generated motions scored twice, not two independent draws — only the ground-truth
reference moved between the runs (102/128 correct in round 1's GT batching, 104/128 in round
2's). Round 2 therefore measured **ground-truth batching variance**, not generation variance, and
did not test the single-run-noise hypothesis at all.

**But that same accidental design produced a sharper, controlled measurement (SUP-20260906-26/27),
holding generation fixed and varying only the reference:**
- **FID moved 1.0731 -> 1.3997 — a +30% swing from redrawing the n=128 reference alone**, with
  the generated set held bit-for-bit identical. A statistic that moves 30% on a same-generation
  re-reference cannot be used to decide a 2x gap against a +/-5% tolerance. **FID at n=128 is
  underpowered — no conclusion is available from it, in either direction.** The earlier framing
  in this entry's "Round 2" section and in `docs/DECISIONS.md` D-22 (treating the FID gap as
  "real, ~2x, not rescued by sample size") overstated what a single small-n comparison can
  support; retracted here, not silently smoothed over.
- **R-Precision moved only +-2/128 ~= 0.016 between the two runs (pure ground-truth batching
  noise) — the generated excess (0.7578 - 0.611 = 0.147) is about 9x that noise floor.** This
  reframes the earlier "~3.5 sigma" estimate (which used the bundled log's cross-replication CI as
  a proxy for this run's noise) with a direct, within-this-experiment noise measurement instead,
  and the conclusion strengthens rather than weakens: **the R-Precision anomaly is real and
  substantially exceeds observed noise, unexplained after every cheap lead checked.**

**The honest split, going forward: FID at n=128 — no conclusion. R-Precision — a real, unexplained
anomaly, ~9x the observed noise floor.** Treat these as two separately-resolved questions, not
one combined "the FAIL is confirmed" or "the FAIL is explained" statement.

**Round 3 (2026-09-06), per review SUP-20260906-28 — the one further E0b action authorised: fix a
reference, not the generation.** D-22 stands: no further diffusion generation. But a full-scale
ground-truth reference costs nothing to build (no diffusion sampling — only encoding real motions
through the small evaluator, which is fast) and removes the reference-redraw variance that rounds
1-2 showed dominates the n=128-vs-n=128 comparison entirely.
`scripts/e0b_fixed_reference_rescoring.py`: materialized the full test split (4198 sequences,
matching E0a's scale), built and saved a fixed `(mu, cov)` reference from it
(`../artifacts/e0/fixed_gt_reference/fixed_gt_reference.npz`, n=4640 after
`Text2MotionDatasetV2`'s sub-clip splitting — reusable for every future FID in this project), then
re-scored the **same 128 cached generated motions, zero regeneration**, against this fixed
reference.

**Result: FID = 3.2909 — higher than either round 1 (1.0731) or round 2 (1.3997), not lower.**
This was not the expected direction, and it is informative rather than a failure of the method:
fixing only the *reference* side does not fix FID's small-n instability, because the *generated*
side is still only n=128 — a 512-dimensional covariance estimated from 128 samples is severely
rank-deficient (rank <=127) regardless of how well-estimated the other side is. Pairing a
well-conditioned, full-scale reference covariance against a poorly-conditioned n=128 test
covariance is its own source of Frechet-distance instability, distinct from (and evidently at
least as large as) the instability from having both sides small. **FID cannot be stabilized at
this generated-sample-count by fixing the reference alone; both sides need adequate n.** This is
consistent with, and sharpens, the "FID at n=128 is underpowered" conclusion above — it is
underpowered specifically because of the *generated* set's size, not the reference's.

**Next:** D-03 remains UNRESOLVED; downstream numbers are internally-comparable-only until it
resolves — this is not a resolved-but-unfavourable result, it is genuinely undetermined at the
sample sizes this hardware can afford. R-Precision is the one number from E0b that stands as a
real, unexplained finding (not sample-size noise); FID from E0b supports no conclusion at all.
Per D-22, no further generation is planned to close this. The one hypothesis that remains
formally untested (not planned, per D-22): whether this project's 128-sample subset (first rows
in HF streaming order) is representative of the full test set the reference protocol draws from
— testing this would need either a properly randomized subset draw at the same n, or accepting the full
n~1000 scale (~5 CPU-hours) to remove the question entirely. Holding before spending either,
to report the exhausted-leads status first. Still open, lower priority: fix the
`diversity_times` off-by-one (now free to do without regeneration, since motions are cached).

---

## E1A-power — pilot power check, PRE-REGISTERED before running (review SUP-20260906-33)

**Ran by:** `../scripts/e1a_power_check.py`   **Date:** 2026-09-06   **Seeds:** 10 (MDM default, single seed — this is a power check, not the E1A result itself)
**Data:** HumanML3D train-split materialized subset for training, test-split materialized subset
for evaluation (`third_party/motion-diffusion-model/dataset/HumanML3D/`) — **corrected before
running (review SUP-20260906-37)**, see note below.
**Record:** `../artifacts/e1/e1a_power_check_record.json` (to be written by the run)
**Status:** PRE-REGISTERED, not yet run

**CORRECTION, before any run happened (review SUP-20260906-37, 2026-09-06):** this entry
originally specified training and evaluating on the same materialized subset (test — the only
split on disk at the time), with the limitation stated in "Does NOT establish" below as
"necessary but weaker" than held-out generalization. **That framing was wrong, not just weaker:**
a model can score above the 0.09375 chance threshold purely by memorising which of ~4,648
training captions pairs with which training motion, which is exactly what evaluating on the same
split would measure — the gate could pass for a reason unrelated to the question it exists to
answer (does this budget teach generalisable text conditioning), disabling the check rather than
weakening it. Fixed by materializing a real, disjoint HumanML3D train subset
(`materialize_humanml3d_test_subset.py --split train`, train ids prefixed `train_sample######`
so they cannot collide with the existing test files) and rewiring `e1a_power_check.py` to train
on `split=train`, evaluate on `split=test`. **The hypothesis and the 0.09375 success criterion
below are unchanged** — only the split wiring was wrong, not the gate itself. Original text below
left as originally written, per the append-only/correction convention, with this note governing.

**Why this run exists, before any A-vs-B comparison:** the director (review pass, SUP-20260906-33)
pointed out that E1's proposed budget (3,000 training steps) is 0.63% of MDM's published
475,000-step budget — both E1A and E1B would be severely undertrained. If neither arm has learned
to use text conditioning at all, they will score identically not because caption truncation is
harmless, but because neither arm can exploit a caption in the first place — a floor effect that
would be misread as "no measurable difference" when the real finding is "this budget has no power
to detect the effect." Running E1A alone first, cheaply, checks for this before spending the full
matrix's wall-clock on an experiment that cannot conclude either way.

**Hypothesis (pre-registered):** at 3,000 training steps (single seed, real MDM `trans_enc`
architecture, `docs/DECISIONS.md` D-23/D-24 config), the trained model's generated motions score
above chance on R-Precision-top3 against their own captions — i.e., the model has learned
*something* about the text-motion relationship at this budget, not zero.

**Success criterion (pre-registered), decided before the run per `docs/DECISIONS.md` D-25's
R-Precision-decisive regime:** chance-level R-Precision-top3 over a 32-candidate retrieval pool
is **3/32 = 0.09375**.
- **R-Precision-top3 clearly above 0.09375** (a margin larger than the measured ground-truth
  batching noise floor of ~2/128 ≈ 0.016, per E0b) → the pipeline has learned text conditioning at
  this budget; E1 has power; proceed to E1B and additional seeds.
- **R-Precision-top3 at or near 0.09375** → this budget is below the threshold where E1 can
  resolve anything; **stop** rather than run the full matrix. The correct next step is then a
  design decision (more steps, a smaller/faster model, or an explicit "E1 is not affordable at a
  budget that gives it power on this hardware" result — itself a legitimate, honestly-labeled
  finding about what a laptop-scale rebuild can and cannot establish), not a silent re-run at a
  bigger budget presented as if it were always the plan.

| metric | value | seed spread |
|---|---|---|
| R-Precision-top3 (generated, n=128) | *(to be filled in)* | n=1, no spread — single-seed pilot only |
| R-Precision-top3 (ground truth, n=128, same batches) | *(to be filled in)* | reference point, not a comparison target |
| FID (generated vs. fixed reference) | *(to be filled in, secondary only per D-25 — not used to decide this check)* | n=1 |

**Result:** *(to be filled in after the run — this entry is written before training starts,
per the pre-registration discipline used for E0b and E2)*
**Establishes:** *(to be filled in)*
**Does NOT establish:** Trained and evaluated on the same materialized subset (HumanML3D's real
train split is not yet materialized on disk) — a positive result here shows the architecture *can*
exploit text at this budget when the eval captions were also seen in training, which is a
necessary but weaker condition than generalizing to held-out captions. This check answers "is
there enough training signal for E1's comparison to have any power at all," not "does this model
generalize." Single seed — no seed-to-seed spread is measured by this pilot; E1's own seed
handling comes at the next stage if this gate passes.

---

## E1-pilot — zero-training caption-truncation information-loss check, PRE-REGISTERED before running (review SUP-20260906-34)

**Ran by:** `../scripts/e1_pilot_caption_truncation.py`   **Date:** 2026-09-06   **Seeds:** 10 (single seed — no model, no stochastic generation involved, so seed only controls DataLoader shuffling)
**Data:** HumanML3D test-split materialized subset, real motions and real captions/tokens — no synthetic data
**Record:** `../artifacts/e1/e1_pilot_caption_truncation_record.json` (to be written by the run)
**Status:** PRE-REGISTERED, not yet run

**Why this runs before E1B, and before the E1A power check finishes:** the director (review
SUP-20260906-34) pointed out that E1's already-validated evaluator (ground-truth R-Precision
reproduced to 0.06σ of a 20-replication reference, per E0b) can measure the caption-truncation
half of F3's defect directly — same real motions, full captions vs. the original project's own
truncation rule applied to the same captions, no model, no generation, minutes instead of hours.
This is an **upper bound on E1B**: a trained model conditioned on truncated captions cannot
recover information the truncation already destroyed, so whatever drop this measures is at least
as large as what E1B could ever show, and probably larger (E1B's undertrained model attenuates
the effect further, per the E1A-power entry above).

**Hypothesis (pre-registered):** replacing full captions with the original project's
first-action-clause truncation, on the *same* real test motions, measurably lowers R-Precision-top3
in the validated evaluator's embedding space.

**Success criterion / decision rule (pre-registered):**
- **Drop is large** (well above the measured ground-truth batching noise floor of ~2/128 ≈ 0.016,
  per E0b) → caption truncation destroys real, substantial text-motion alignment signal; E1B is
  worth its ~7.5h training cost, and this number is a pre-registered prediction of roughly what
  scale of effect E1B should find (attenuated further by undertraining, not amplified).
- **Drop is near zero** → the truncation rule, as the original project actually implemented it,
  does not remove meaningfully retrievable text-motion signal in this embedding space; **E1B can
  be dropped or deprioritized**, and F3's caption-truncation half is answered in minutes rather
  than hours. This would not mean F3 is wrong (F3's own pose-dispersion evidence stands
  independently) — it would mean the *caption* side of the conditioning-mismatch story is weaker
  than the *frame-selection* side, which the same logic cannot cheaply test (see below).

**Scope, stated before running:** this measures information loss in the text encoder / retrieval
space, on real motions. It does **not** establish that a model trained on truncated captions
generates worse motion — only that truncation removes (or does not remove) N points of
retrievable text-motion alignment. It is a predictor of E1B's effect size, not a substitute
result for E1B itself.

**Asymmetry, stated rather than papered over (per review):** there is no equally clean
zero-training analogue for E1C (frame-selection). Comparing a frame-0-replicated static motion
against real motion in the same retrieval space would reintroduce exactly the "static motion is
unusual to a sequence-trained evaluator" confound that SUP-30 removed from the generation-based
design. E1C stays a generation-based check only; this pilot does not extend to it.

| metric | value | notes |
|---|---|---|
| R-Precision-top3, full caption (same motions) | **0.8013** | vs. E0b's ground-truth 0.7969 — 0.0044 apart, well inside the ~0.016 noise floor; validates this run's pipeline matches E0b's |
| R-Precision-top3, truncated caption (same motions) | **0.6563** | |
| Drop | **0.1450** | ~9x the 0.016 noise floor — large, not noise |
| Matching Score (lower is better) | full 2.985 → truncated 3.895 | same direction, corroborates the R-Precision drop independently |
| % of captions using the original's "first sentence" fallback vs. its conjunction-tag branch | **44.8% fallback** (5,621 / 12,542), mean words 12.62 → 8.01 | the conjunction-tag branch (CCONJ/SCONJ) did fire for the majority (55.2%) — the suspected-dead `/ADV then\|after\|before` literal-substring sub-branch was not separately instrumented, so this run cannot say whether *that specific* sub-branch ever fired; the CCONJ/SCONJ path alone is sufficient to explain most non-fallback cases |

**Result:** Hypothesis confirmed, decisively. Caption truncation to the original project's own
first-action-clause rule drops R-Precision-top3 by 0.145 (0.8013 → 0.6563) on the *same* real
motions, ~9x the measured batching noise floor. This is not a small or marginal effect — it is a
large, clean signal that the original's truncation rule destroys substantial text-motion
alignment information, independent of any model, training budget, or generation step. The
full-caption arm's 0.8013 sits within noise of E0b's independently-measured 0.7969, confirming
this run's evaluator/pipeline matches the already-validated one rather than measuring something
different.
**Establishes:** The caption-truncation half of F3's conditioning-mismatch defect is real and
large in the evaluator's own embedding space — a genuine information-destruction effect, not an
artifact of small-n noise (`LANDMINES.md` §13/§14 concerns do not apply here: this is
R-Precision, stable at this n, and the two arms are compared on identical real motions in the same
run). **Per the pre-registered decision rule: this is a large drop, so E1B remains worth its
~7.5h training cost**, and 0.145 is now a pre-registered prediction of the rough scale E1B should
find — expected to be *attenuated*, not amplified, once the effect is filtered through an
undertrained generative model (per the E1A-power entry above), so E1B finding a smaller drop than
0.145 would not be surprising; E1B finding a *larger* drop, or a drop in the opposite direction,
would be the more informative outcome and should be flagged as such rather than absorbed quietly.
**Does NOT establish:** Generation quality under truncated conditioning — this is a text-encoder/
retrieval-space measurement only; E1B (if it proceeds) still measures something this pilot cannot:
whether a trained generative model's *output* degrades, not just whether the caption itself
carries less retrievable signal. Anything about the frame-selection (E1C) half of the defect — no
analogous zero-training check exists for it, per the asymmetry note above. Whether the specific
`/ADV then|after|before` literal-substring sub-branch of the original's regex ever fires in
practice (this run only distinguishes "any conjunction/adverb branch matched" from "fell through
to first-sentence fallback," not which literal alternative matched).

---

### E1-pilot follow-ups — length-matched control (SUP-20260906-38) and non-fallback-subset conditional effect (SUP-20260906-39)

**Ran by:** `../scripts/e1_pilot_followups.py`   **Date:** 2026-09-06   **Seeds:** 10
**Data:** same HumanML3D test-split subset as the E1-pilot above. One determinism change from
that run, stated plainly: this script fixes each motion to its **first** text entry rather than
`Text2MotionDatasetV2`'s own `random.choice` per draw, because SUP-39's subgroup restriction needs
a stable fallback/non-fallback label per key. This is why this run's own full/truncated numbers
(0.8116/0.6552) are close to but not bit-identical to the original E1-pilot's (0.8013/0.6563) —
both are reported, neither substituted for the other.
**Record:** `../artifacts/e1/e1_pilot_followups_record.json`
**Status:** VERIFIED — both are real retrieval re-runs, not arithmetic estimates from the
original aggregate numbers.

**SUP-38's question:** is the 0.145 drop caused by *which words the original's rule selects*, or
just by *captions getting shorter*? Control: truncate the same full captions to the same
per-caption word count via a content-neutral first-N-words rule, re-measure.

**SUP-38 result — the effect is a length effect, not a rule-specific one.** Length-matched
control R-Precision-top3 = **0.6468**, essentially the same as the rule-truncated arm's **0.6552**
(this run's own recomputation) — the rule-truncated arm scored *very slightly higher*
(`rule_specific_drop_beyond_length` = -0.0084, i.e. the "smart" conjunction-based truncation was
marginally less damaging than blind first-N-words truncation, not more). **This refines, not
negates, the E1-pilot's headline finding:** caption truncation genuinely destroys real
text-motion alignment signal (confirmed again here: corpus-wide drop recomputed at 0.1565,
consistent with the original 0.1450) — but the *mechanism* is caption length/information density,
not something specifically bad about the original project's clause-selection heuristic. A naive
"take the first N words" truncation would have done comparable damage. This matters for how the
finding should be described going forward: "truncating captions destroys alignment signal" is
supported; "the original's specific rule for choosing where to cut is uniquely bad" is not.

**SUP-39's question:** does the corpus-wide 0.145/0.157 drop understate the effect where the
rule actually fires, since 44.8% of captions fell through to a near-harmless first-sentence
fallback and dilute the average?

**SUP-39 result — yes, substantially.** Restricting to the 2,599 keys (55.9% of the corpus, not
44.8% — the run's own fallback/non-fallback split, computed directly, not assumed) where the
rule actually truncated at a conjunction tag: full-caption R-Precision-top3 = **0.8036**,
truncated = **0.5312**, **drop = 0.2724** — compare the corpus-wide 0.1565/0.1450. Real
measurement, not the director's back-of-envelope `0.145/0.552≈0.263` estimate, and it lands close
to that estimate anyway (0.272 vs 0.263), which is itself a useful cross-check that the linear
approximation wasn't misleading here. **The mechanistically meaningful number for "what does this
rule do when it actually fires" is 0.272, not the corpus-wide 0.145-0.157** — the corpus-wide
number describes what the original project's data actually suffered on average (a fair number to
quote for "how bad was this in practice"); 0.272 describes the rule's effect size where it applied
at all (the fairer number for "how destructive is this kind of truncation," which is also more
directly comparable to SUP-38's length-matched control finding, since non-fallback captions are
the longer, more heavily truncated ones).
**Establishes:** Two real numbers, not one: **0.145-0.157 corpus-wide** (what the original
actually suffered, averaged over its whole caption distribution) and **~0.27 conditional** (what
happens specifically where truncation fired, roughly double the corpus-wide figure and the
number that should anchor any comparison to E1B if it proceeds). Separately, that the mechanism is
caption shortening in general, not the original's specific clause-boundary heuristic — a real,
if less flattering-to-the-original-narrative, refinement.
**Does NOT establish:** Whether the length-matched control finding (effect is length-driven) also
holds specifically *within* the non-fallback subset — this run did not compute a length-matched
control restricted to just the 2,599 non-fallback keys, so it cannot rule out that some of the
0.272 conditional effect is content-selection-specific after all, even though the corpus-wide
result suggests otherwise. Whether either effect propagates into a trained generative model's
output — that is still E1B's question, unaddressed by any zero-training check.
