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
**Power check (pre-registered, added per review SUP-20260906-55 — E1B ran without this and the
gap turned out to be 0.80σ, unresolvable, discovered only after ~5 hours of compute):**
```
Expected effect size:           <from a prior measurement, or labelled a guess>
Measurement SE at planned n:    <computed>
Minimum detectable effect (3σ): <computed>
Verdict: powered / underpowered — and if underpowered, why run it anyway
```

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
**Record:** `../artifacts/e1/e1a_power_check_record.json` (hand-assembled after a crash, see Result below)
**Status:** VERIFIED — gate passes (R-Precision-top3 0.2969 vs. 0.09375 chance, see Result below)

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
| R-Precision-top3 (generated, n=128, held-out test captions) | **0.2969** | n=1, no spread — single-seed pilot only |
| R-Precision-top3 (ground truth, n=128, same batches) | **0.7950** | reference point, matches prior ground-truth measurements (0.7969-0.8036) to within noise |
| FID (generated vs. ground truth) | **7.2093** (secondary only per D-25 — not used to decide this check; expected to be poor at this budget) | n=1 |

**Result: GATE PASSES, decisively.** R-Precision-top3 = 0.2969 against the pre-registered
chance threshold of 0.09375 — **3.17x chance**. Margin = 0.2031. **Corrected (review
SUP-20260906-49): against E0b's established ground-truth batching-noise floor (0.016, the
project's standard reference for R-Precision decisions), that margin is 0.2031/0.016 ≈ **13x**,
not the "~18x" first written here** — the 18x figure was an arithmetic slip (it matches
0.2031/0.0117, a different floor measured in a different comparison — the E1-pilot's
could-vary-subset position-effect check — not the one named in the sentence). Stating which
floor is used and why, per the director's own instruction after catching exactly this kind of
silent floor-substitution elsewhere in this arc: **13x the 0.016 E0b ground-truth batching
floor** is the correct, checked figure for this gate. Not a close call either way. Trained on the disjoint 4,435-sequence materialized train split,
evaluated on the materialized test split's held-out captions — an above-chance result here
cannot be memorisation (per the SUP-20260906-37 fix), since the model never saw these
caption-motion pairs during training. Training took 6,877.9s (1.911h) for 3,000 steps; generation
took 2,626.9s (0.730h) for 128 samples — both close to the feasibility probe's projections
(1.877h / 0.65h). **The script crashed on the known `diversity_times` off-by-one bug (same as
E0b, `assert activation.shape[0] > diversity_times` with `diversity_times=128=num_samples_limit`)
after R-Precision/FID/Matching-Score had already printed** — record hand-assembled from the run
log (`artifacts/e1/e1a_power_check_record.json`), same precedent as E0b's own crash-recovery.
Log-space convergence fraction (SUP-20260906-42, context only, not decisive): using the final
logged step's loss (0.19136) against the converged reference (0.0563) and the untrained-init loss
(1.36502), `log(L_init/L_obs)/log(L_init/L_conv) ≈ 0.62` — roughly two-thirds of the way from
init to convergence in log-space. Consistent with, not contradicting, the R-Precision gate: both
signals agree the model learned real structure at this budget, and per SUP-42's own caveat this
loss number is reported as context, not as what decided the gate.
**Establishes:** at 3,000 training steps on ~4,400 disjoint train sequences, MDM's real
architecture learns genuine, generalisable text-motion conditioning — not memorisation, not zero.
**E1 has power at this budget.** Per the pre-registered decision rule, E1B (and any additional
seeds) may proceed.
**Does NOT establish:** Generation *quality* — FID=7.2093 and the visibly poor loss trajectory
confirm the model is, as expected, far from a usable generator at this budget (D-24's point that
3,000 steps is 0.63% of MDM's published training length). This check only establishes that text
conditioning is being learned at all, which is the minimum condition for E1B's A-vs-B comparison
to mean anything — it does not establish that E1B's eventual result will be clean or large.
Single seed — no seed-to-seed spread measured here; E1's own multi-seed handling is a separate,
still-open design question for whichever arms actually run.

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
number that should anchor any comparison to E1B if it proceeds). **Corrected below (SUP-44):**
this entry originally concluded "the mechanism is caption shortening in general, not the
original's specific clause-boundary heuristic" from the length-matched control alone — that
control (naive first-N-words) keeps the caption *prefix*, same as the rule being tested, so it
could only show "among prefix-preserving rules, the cut point doesn't matter," not "position is
irrelevant." The random-window follow-up below closes that gap.
**Does NOT establish (as originally written; see the SUP-44 follow-up immediately below for what
closes this):** Whether the length-matched control finding (effect is length-driven) also
holds specifically *within* the non-fallback subset — this run did not compute a length-matched
control restricted to just the 2,599 non-fallback keys, so it cannot rule out that some of the
0.272 conditional effect is content-selection-specific after all, even though the corpus-wide
result suggests otherwise. Whether either effect propagates into a trained generative model's
output — that is still E1B's question, unaddressed by any zero-training check.

---

### E1-pilot follow-up 2 — random-window control separates length from position (review SUP-20260906-44)

**Ran by:** `../scripts/e1_pilot_followups.py` (extended with a fourth arm)   **Date:** 2026-09-06
**Record:** `../artifacts/e1/e1_pilot_followups_record.json` (same file, re-run with the new arm added)
**Status:** VERIFIED

**The gap this closes:** both prior controls (the original's rule, and the length-matched
first-N-words control) keep the caption's *prefix* — the rule cuts at the first conjunction,
which sits near the front; first-N-words cuts at the front by construction. Neither could
distinguish "shorter captions retrieve worse" from "keeping the front of the caption matters
specifically." Added a random contiguous N-word *window* (same per-key length, different
position, drawn once with a fixed seed) as the control that actually varies position.

**Result — a clean null on position; the length-driven conclusion now holds properly.**
Random-window control R-Precision-top3 = **0.6470**, statistically indistinguishable from both
the length-matched prefix control (0.6468) and the rule-truncated arm (0.6552).
`position_effect_prefix_minus_random_window` = **0.00022** — effectively zero, nowhere near the
noise floor threshold that would make it interesting. There is no detectable advantage to keeping
the front of a HumanML3D caption over keeping an arbitrary same-length slice of it.
**Caveat, stated directly:** 1,985 of the 4,648 keys (42.7%) had no room to move (the caption was
already short enough that the "random window" and "first N words" are the same slice by
construction) — the position manipulation is only meaningfully exercised on the remaining 2,663
longer captions. Reported as a limitation on how much this specific run could vary position, not
as a reason to discount the null result, since even among captions long enough to vary, the
aggregate score barely moved.

**Re-slice on the could-vary-only subset (review SUP-20260906-45): the null holds up under real
restriction, not just dilution-corrected arithmetic.** Restricting both position controls to
exactly the 2,663 keys that had room to place a different window (an actual re-run of
`evaluate_matching_score` on the restricted subset, not the director's own
`0.00022/0.573≈0.00038` linear estimate, though that estimate is what a purely-diluted-by-
construction null would have produced): length-matched-prefix = **0.5456**, random-window =
**0.5339**. `position_effect_on_could_vary_subset_only` = **-0.0117** (computed as
random-window minus prefix, so the prefix control scored *higher*).

**Sign correction (review SUP-20260906-46, and my own error, not the director's second time
around): this entry originally called the prefix-scores-higher direction "opposite [of] a
front-loading hypothesis."** That is backwards. Front-loading predicts exactly this direction —
if HumanML3D captions put motion-relevant content near the front, a rule that *keeps the front*
should retain more retrievable signal than one that keeps an arbitrary middle span, i.e. prefix
should score higher. **The point estimate (+0.0117 favoring the prefix) is directionally
consistent with front-loading, not a refutation of it — the correct reading is "not enough
precision to resolve a position effect," not "no position effect."**

**Local noise-floor check (also SUP-20260906-46 — the earlier "within the ~0.016 floor" claim
borrowed that number from E0b's n=128 ground-truth batching check, a different n and a different
absolute-score range; this run measures the floor actually local to this comparison instead of
borrowing one).** Re-shuffled the same restricted (n=2,663) datasets with a second seed:
length-matched-prefix moved 0.5456 -> 0.5448 (**0.0008** swing), random-window moved
0.5339 -> 0.5215 (**0.0124** swing) — batching noise at this n/score range is real and, for the
random-window arm specifically, close in magnitude to the measured "position effect" itself
(0.0117). **Correct conclusion: no position effect resolvable at this precision; the point
estimate leans the way front-loading predicts but sits within this comparison's own measured
noise band, so this is a statement about power, not about the absence of an effect.**
**Establishes (superseded by the averaged result immediately below — left here per the
append-only convention rather than deleted, since it was the correct read of a single draw):**
at the time this was written, with one random-window placement draw, the position effect looked
underpowered-to-detect rather than resolvable.

---

### E1-pilot follow-up 3 — averaging the random-window arm over 8 placement draws resolves a real position effect (review SUP-20260906-47)

**Ran by:** `../scripts/e1_pilot_random_window_averaged.py`   **Date:** 2026-09-06
**Record:** `../artifacts/e1/e1_pilot_random_window_averaged_record.json`
**Status:** VERIFIED

**Why the single draw was the wrong instrument:** the director pointed out that the prefix
controls are deterministic given a target length — reseeding only changes which captions land in
which retrieval batch. The random-window control is different: reseeding changes the *placement
itself*, a different treatment each time, not a different sample of the same treatment. Its
single-draw batching-noise swing (0.0124, measured in follow-up 2 above) was therefore mostly
**treatment variance** — one arbitrary placement's outcome — not measurement noise, and "random
scored about the same as prefix" from one draw could not distinguish "position doesn't matter"
from "this particular placement happened to land close to the prefix."

**Method:** 8 independent placement draws (different `random.seed` per draw, same 2,663 could-vary
keys, same per-key window length, same evaluator), with the DataLoader batch-shuffle seed held
fixed across all 8 draws so only placement varies — isolating treatment variance from the
batching variance already characterized separately in follow-up 2.

**Result — the effect is real once treatment variance is averaged down.** Mean R-Precision-top3
across 8 draws = **0.5279** (std across draws = 0.0082, min 0.5132, max 0.5384 — real,
substantial placement-to-placement variance, confirming the director's diagnosis that a single
draw was not a stable estimate). Standard error of this 8-draw mean ≈ 0.0082/√8 ≈ **0.0029**.
Compared against the length-matched prefix control (0.5456, itself stable within 0.0008 across
batch-shuffle seeds): **gap = 0.0177, roughly 6× the standard error of the averaged mean** — a
materially resolved effect, not noise. **Flagged loudly per the director's own instruction, since
this reverses follow-up 2's "underpowered to detect" read:** with the treatment-variance problem
fixed, the prefix control reliably outperforms the *average* random placement by ~0.018
R-Precision-top3 points on this 2,663-key subset.
**Establishes:** the front-loading hypothesis is **supported, not merely "not refuted"** — keeping
the caption prefix retains measurably more retrievable text-motion alignment signal than an
average arbitrary same-length window. HumanML3D captions front-load motion-relevant content to a
real, if modest (~0.018 of ~0.15-0.27 total truncation cost), degree. **Revised final position on
the whole E1-pilot arc:** truncation cost is driven overwhelmingly by *how much* text is removed
(the corpus-wide 0.145-0.157 and conditional ~0.27 numbers, both far larger than this 0.018), with
a small, now-resolved, secondary contribution from *which part* — the front of a HumanML3D caption
carries slightly more than an average middle span, but not by enough to change which mechanism
dominates the original project's actual truncation cost.
**Does NOT establish:** whether averaging over multiple batch-shuffle seeds too (not just
placement draws) would move this further — this run held the batch-shuffle seed fixed
deliberately, at the director's own instruction, to isolate placement variance cleanly; the
~0.0008-0.0124 batching-noise band from follow-up 2 is a separate, smaller source of uncertainty
not re-combined with this estimate's own error bar here. A generalizable methodological note
added to `LANDMINES.md` (see below): when one arm of a comparison is itself stochastic, its
across-seed spread is treatment variance, not measurement noise, and must be averaged down before
that arm is comparable to a deterministic one.
**Does NOT establish:** Whether this generalizes to a different embedding space or a different corpus's
caption style. Anything about generation — still entirely E1B's open question.

---

### E1-pilot closing synthesis (review SUP-20260906-48)

**Statistical confirmation, independently checked:** gap 0.0177 against the combined standard
error of the prefix arm (stable to 0.0008 across seeds) and the averaged random-window arm
(SEM 0.0029) is **5.9σ**, not the 6× figure reported in follow-up 3 (which omitted the prefix
arm's own, much smaller, uncertainty). The difference is immaterial to the conclusion — front-
loading is confirmed either way — but the more careful number is the one on record.

**Decomposition of the conditional truncation cost (~0.27, the number where the rule actually
fires, from the original follow-up above):** volume (how much text is removed) accounts for
roughly **93%** of it; position (which part is removed) accounts for roughly **7%**
(0.0177/0.27) — real and statistically resolved, but a small secondary contributor next to the
dominant mechanism. **"Truncation cost is driven overwhelmingly by how much text is removed, with
a small but statistically resolved contribution from which part"** is the single sentence that
best summarizes the whole arc, and is stronger than either the bare null this pilot could have
stopped at, or the "the original's rule is uniquely bad" story it opened with.

**Why this is a real effect and not one manufactured by iterating until something moved:** worth
stating explicitly rather than leaving for a reader to infer from the sequence of entries above.
(1) Every refinement across all three follow-ups was proposed on a priori methodological
grounds — undiluting a stratum that mechanically cannot express the manipulation (follow-up 1),
averaging an arm whose seed controls the treatment rather than the sample (follow-up 3) — never
because a prior result looked wrong and needed to be pushed further. (2) The hypothesis direction
was on the record in advance, twice, before this measurement existed: front-loading (prefix scores
higher) was named as the alternative outcome in the original SUP-38 request and again in SUP-44's
sign correction. The final result confirms a prediction that pre-dated it, rather than explaining
a surprise after the fact. **The one real caveat:** the specific comparison that resolved this —
an 8-draw-averaged random arm against a deterministic prefix, restricted to the 2,663-key
could-vary subset — was itself assembled across follow-ups in response to review, not
pre-registered as a single design before any of this pilot ran. Stated plainly; it costs nothing
to say and would cost more to leave implicit.

**The review sequence, kept in the record because it is the honest part:** the director's own
three passes on this specific question moved from "refuted" (an overclaim from an underpowered
single draw) to "not resolvable at this precision" (correct given the evidence at that point) to,
after proposing the averaging fix, "confirmed." Neither of the first two reads was right, and they
were wrong in opposite directions — the process of successive, methodologically-motivated
correction is what produced a trustworthy final number, not either party's standalone judgment at
any single point. That sequence is preserved across follow-ups 2 and 3 above rather than
summarized away here.

**E1-pilot is closed.** Two findings survive: a large, robust volume effect (corpus-wide
0.145-0.157, conditional ~0.27) and a small, statistically resolved position effect (~0.018, ~7%
of the conditional cost). The original project's first-action-clause segmentation — presented as
a deliberate design choice with a POS-tagging algorithm behind it — is shown to be neither clever
nor uniquely harmful: one of many ways to discard roughly a third of a caption's words, and nearly
all of the cost came from the discarding itself, not from where the cut was made.

---

## E1B — truncated caption, trained and evaluated end-to-end, RESOLVED as an affordability finding (0.80σ, D-26) (director go-ahead after E1A's gate passed)

**Ran by:** `../scripts/e1_train_arm.py --arm b`   **Date:** 2026-09-06   **Seeds:** 10 (single seed — seed spread not yet established, see below)
**Data:** HumanML3D train-split materialized subset (4,435 sequences) for training, test-split
materialized subset for generation-conditioning and evaluation — disjoint, per D-25/SUP-37.
Captions truncated via the same faithful first-action-clause rule used throughout the E1-pilot,
applied to BOTH training and generation-conditioning captions (design decision recorded in
`LEDGER.md` Item 30).
**Record:** `../artifacts/e1/e1b_train_record.json`
**Status:** RESOLVED — as a power/affordability finding, not a directional one. See the analysis
below (review SUP-20260906-51/52/53); this supersedes the "PARTIAL, pending E1A seed 2" framing
this entry originally carried. Process note, corrected per SUP-20260906-54 to separate two
things that got conflated: E1B's **hypothesis and success criterion were pre-registered** — both
are in `REBUILD_SPEC.md` §6's ladder row, written before the run, not reconstructed after. What
was actually missing is the **per-run record table** (the filled-in metric/seed-spread table)
that E0b and E1A-power each got committed before their own runs started — a documentation-
consistency gap, not a pre-registration failure. The more consequential gap, per §18's new
landmine entry, is that no minimum-detectable-effect calculation was done before the run — that
is the thing that would actually have flagged the affordability ceiling in advance.

**Raw result:** R-Precision-top3(E1B) = **0.3438**, against E1A's **0.2969**. Ground-truth
R-Precision-top3 = 0.7950 (consistent with every prior measurement). FID(E1B) = 8.3402 (worse
than E1A's 7.2093, secondary metric per D-25). Training took 6,819.5s (1.894h), generation
2,262.3s (0.628h) — both close to E1A's timing.

**The raw number goes the OPPOSITE direction from the pre-registered prediction.** §6's criterion
expected E1B *worse* than E1A. E1B's raw R-Precision-top3 is *higher*. Reported exactly as
measured — no adjustment, no re-running with a different seed to see if it "corrects itself."

**Why this gap does not license any directional claim — the analysis that closes this rung
(review SUP-20260906-51, independently re-derived before being accepted, not taken on faith):**
R-Precision-top3 at n=128 is a proportion of a binomial count, and the raw counts behind these
percentages are small: E1A = 38/128, E1B = 44/128 — **the entire "effect" is 6 samples.**

```
p(E1A) = 0.2969, SE = sqrt(0.2969 * 0.7031 / 128) = 0.0404
p(E1B) = 0.3438, SE = sqrt(0.3438 * 0.6563 / 128) = 0.0420
gap = 0.0469, combined SE = sqrt(0.0404^2 + 0.0420^2) = 0.0583
z = 0.0469 / 0.0583 = 0.80 sigma
```
0.80 sigma is not a signal by any normal standard (2-3 sigma minimum for any directional claim).
**This is binomial sampling noise on the generated-sample count alone — before any training-seed
variance (initialization, data order, generation stochasticity) is added on top, which would only
widen this further, never narrow it.** This single calculation is why the earlier "run E1A seed 2
to get a seed spread" plan (`LEDGER.md` Item 35) is superseded, not merely satisfied early: seed
variance cannot rescue a comparison where the binomial floor alone already exceeds the observed
gap. Cost to resolve at 3 sigma (proportionally scaling n by the square of the SE-reduction
needed): **roughly 1,780 generated samples per arm per seed — about 9 CPU-hours of generation
alone, per arm, per seed**, on top of training time. Not affordable on this hardware, not close.
**Correct statement, and the distinction matters (per SUP-46's precedent — "not supported" is not
the same claim as "refuted"):** the pre-registered hypothesis (E1B measurably worse than E1A) is
**not supported at this sample size** — this is different from "the opposite is true." No
directional claim of any kind is licensed by this data.
**Establishes:** **E1's generation-side question is not answerable at any sample size this
hardware affords, and that boundary is itself now measured, not assumed** — resolving the
observed 0.047 gap at 3 sigma requires ~1,780 samples/arm/seed, computed directly from this run's
own binomial statistics. This is a legitimate result, pre-registered as a possible outcome before
any of E1 ran (`docs/EXPERIMENT_LOG.md`'s E1A-power entry, quoting SUP-20260906-33: "'E1 is not
affordable at a budget that gives it power on this hardware' — itself a legitimate, honestly-
labeled finding"). Everything else about this rung worked: E1A's power check passed its gate at
3.17x chance, ground truth reproduced at 0.7950 against a 0.7977 reference, training/generation
timing matched projections to within minutes, and the `diversity_times` fix held through a clean
exit. **Only the affordability of the generation-side comparison failed — and the caption-
truncation question itself is already answered, model-free, by the E1-pilot at 9-17x its own
noise floor** (corpus-wide 0.145-0.157, conditional ~0.27, both far above what this generation
comparison could ever resolve at this scale). E1B was only ever testing whether that already-
established retrieval-space effect propagates into generation; that propagation question remains
open, not because the experiment failed, but because answering it costs more than this project's
hardware affords.
**Does NOT establish:** any direction for how caption truncation affects generation quality —
supported or refuted. Whether a larger, unaffordable-here sample size would resolve it one way or
the other. Per D-26 (director's decision, `docs/DECISIONS.md`), the ladder stops at this rung:
E1C, a second seed of A or B, and any further seeds are not run, and are documented here as
"legitimate, unaffordable, and not pursued further" rather than silently dropped.

**Pre-registered prediction (review SUP-20260906-54), written before the E1A seed-2/SUP-49
decomposition control lands, so it is a hypothesis being tested rather than a story fitted to a
number afterward.** The raw gap's direction (E1B scoring higher than E1A) is surprising given the
E1-pilot's own finding that truncation costs real retrieval signal (0.145 on real motions,
R-Precision ~0.80). A candidate explanation for why these need not conflict: **caption
specificity is only an asset when the motion carries enough signal to use it.** Real motion
scores ~0.80 R-Precision; this project's own generator scores ~0.30 — vague, largely generic
output. A short, generic caption may match vague, generic motion about as well as a long,
specific caption does, because the specific caption's extra content has nothing in a
low-signal motion to attach to, and may act as noise in the embedding rather than a useful
signal. **If this holds:** truncation destroys information that helps when the motion carries
enough signal to use it (real motion, high R-Precision) and is neutral-to-helpful when it does
not (this project's own severely undertrained generator, low R-Precision) — a claim about the
*interaction between conditioning specificity and generator quality*, more interesting than a
claim about either arm alone, and testable later at higher generator quality. **Stated now,
before the control's number is known, specifically so it cannot be read as after-the-fact
rationalization if it turns out to match.**

**Landed 2026-09-06 (`../artifacts/e1/e1a_seed2_train_record.json`). Supplementary context only
— does NOT reopen or revise the closed 0.80σ conclusion above (D-26); reported because it was
already running and pre-registered, not because it changes the answer.**

- **Arm A, seed 20:** R-Precision-top3 = **0.34375** (44/128) — close to E1B's 0.3438, well
  above seed-1's 0.2969.
- **SUP-49 decomposition control** (the same seed-20 generated motions, rescored against
  *truncated* captions instead of the full captions they were generated from): R-Precision-top3
  = **0.3125** (40/128) — **lower**, not higher or flat, than the full-caption score.
- **The pre-registered prediction (above) is not supported.** "Neutral-to-helpful" would predict
  the truncated-rescore score should match or exceed 0.34375; it came in 0.03125 lower. Binomial
  check: SE(combined) = 0.0587, z = 0.03125 / 0.0587 = **0.53σ** — like every other gap on this
  ladder, this is noise, not evidence against the prediction either. **The decomposition control
  answers "is any of the A-vs-B gap explained by retrievability alone" with "not measurably, at
  this n" — consistent with D-26, not a new result.**
- **A second, independent confirmation of D-26's own logic, this time empirical rather than
  projected:** seed-1 E1A (0.2969) vs seed-2 E1A (0.34375) — same arm, same training recipe,
  different seed only — gap = 0.0469, z = **0.80σ**. This is, to three significant figures, the
  *same* z-score as the original E1A-vs-E1B comparison (also 0.80σ). Seed-to-seed noise within
  one arm is statistically indistinguishable in size from the cross-arm "effect" this ladder set
  out to measure — not a theoretical projection this time, an observed instance of it. Averaging
  E1A's two seeds ((0.2969+0.34375)/2 = 0.3203) and comparing to E1B (0.3438) narrows the gap
  further, to 0.0235 (z = 0.46σ) — moving *closer* to indistinguishable, not further, exactly as
  D-26 predicted seed averaging would do.
- **Does NOT establish:** any direction for the truncation-generation question, still open per
  D-26. Does not license running E1C or further seeds — the ladder still stops here.

**SUPERSEDED in part (docs/DECISIONS.md D-28, review SUP-20260906-77/79): D-26's "affordably
unresolvable" framing above was itself a defect, independent of hardware.** 1,780 samples/arm was
the n needed to resolve **0.0469** — this run's own noise reading (0.80σ, wrong direction,
reproducible from seed variance alone, as the seed-2 finding above shows directly). Powering an
experiment to resolve its own noise blip is circular: the smaller the blip, the more samples it
"needs." The actual hypothesis-motivated effect size was already measured by the E1-pilot at
0.145-0.157 (retrieval-space truncation cost). Re-derived: n/arm at 3σ for delta=0.157 -> 159,
0.145 -> 187, 0.0469 -> 1,781. **n=128 (what was run) is 70-80% of the n the actual hypothesis
needs, not 7% of it.**

**Restated result:** n=128 establishes a minimum detectable effect of **0.175 at 3σ, 0.117 at
2σ**. Caption truncation's retrieval-space cost does **not propagate to generation R-Precision at
full strength** in this regime: effects >= 0.175 are excluded at 3σ, >= 0.117 at 2σ; whether a
smaller effect (0.05-0.10) exists is open, needing n ~ 400-1,600/arm to resolve. This is a bounded
null, not a clean one — three caveats travel with it: (1) both arms severely undertrained (0.63%
of MDM's budget), well above chance so not a floor artefact, but propagation may need a stronger
generator to manifest at all; (2) retrieval-space and generation-space are different quantities,
so attenuation is expected on theory — "not at full strength" is weaker than "absent"; (3) single
seed per arm for the primary comparison (plus two supplementary A seeds).

**A mechanistic reading (interpretation, not established fact — and corrected before being
recorded anywhere, per SUP-79):** this project's generator scores R-Precision-top3 0.30-0.34. The
right comparison is **MDM's own published, converged, generated score, 0.611±.007**
(`LANDSCAPE.md` line 47) — not the ground-truth row (0.797), which an earlier draft of this
reasoning used in error. Even MDM's real, converged model sits well below ground truth on this
metric (a known field-wide saturation effect, `LANDSCAPE.md` §1.3); this project's model, at
0.63% of that budget, is coarser still. R-Precision at that quality is plausibly driven by gross
features (locomotion, speed, seated) that a first-action-clause truncation *preserves* — if so,
the null is the expected result at this regime, not a failure to detect, and a stronger generator
(where finer distinctions become resolvable) is what would test it.

**Regime-scoping note:** D-25 gates E1/E2 on R-Precision because it survives small n, correct in
this low-quality regime. `LANDSCAPE.md` §1.3 found R-Precision saturated at the published
frontier and recommended FID there instead — if this project ever trains toward that frontier,
the gate must revisit FID before trusting any convergent-model quality claim.

**Decision (docs/DECISIONS.md D-28): E1 is CLOSED at n=128. No new arms, no new seeds, no E1C —
regardless of MPS affordability (D-27 makes n~400-1,780/arm cheap, ~2-3 MPS-hours, but a tighter
bound on this specific forensics question is not worth further compute right now).** A proposal
to run n=384 x 2 arms x 2 seeds was made, independently verified as correctly powered, and then
retracted before anything was launched — recorded here so the retraction is visible next to the
proposal it retracts, not silently absent.

## E-side finding — the standard text-motion evaluator does not generalize from 32-candidate to full-corpus retrieval (review SUP-20260906-58/62/63)

**Ran by:** `../demo/measure_self_retrieval.py`   **Date:** 2026-09-06   **Seeds:** 10
**Data:** 300 real HumanML3D captions sampled from the materialized corpus, real motions,
`EvaluatorMDMWrapper`'s text encoder (the same instrument this project's own R-Precision numbers
come from)
**Record:** `../artifacts/demo/self_retrieval_record.json`
**Status:** VERIFIED — a side finding from Stage 5 demo work, not part of the E-series ladder,
recorded here because it is a property of the evaluation instrument itself, not of this project's
model or dataset.

**Hypothesis (pre-registered, informally — this was a "does this component generalize" check
before shipping a demo feature, not a numbered rung):** the evaluator's text encoder, being the
same validated instrument behind this project's R-Precision numbers, should support full-corpus
nearest-neighbour retrieval (finding a caption's own true motion among all ~8,198 corpus motions,
not just a 32-candidate batch) at least roughly as well as a simple TF-IDF baseline.

**Result: the hypothesis is false, and the reason is diagnosable, not mysterious.** Full-corpus
self-retrieval in the evaluator's embedding space collapsed to **~1%** (two independent samples:
this project's own run and a separately-run comparison, both far below TF-IDF's **74.7-78.3%**
on the identical task). Diagnosed directly: a real caption's own true motion scores **0.978**
cosine similarity against its own text embedding, yet ranks **264th of 8,198** candidates,
because every corpus motion occupies a tight **0.97-0.99** cosine band at full-corpus scale in
this embedding space.

**Establishes:** the text encoder was trained to discriminate a caption from ~31 distractors
(R-Precision's own batch-of-32 protocol) and does not generalize to ranking against thousands of
candidates — a non-obvious property of an evaluation instrument the entire text-to-motion field
shares (this is Guo et al.'s own evaluator, used across MDM, MotionDiffuse, and every number in
`LANDSCAPE.md`'s published-numbers table), not specific to this project's checkpoint or corpus.
This explains *why* the field's own protocol specifies N=32 rather than that being an arbitrary
convention: batch-of-32 is close to the largest pool size this embedding space's resolution
actually supports. Retroactively strengthens `LANDMINES.md` §13 (candidate-pool size is not a
configuration detail, it is the range the instrument is calibrated over at all) with a second,
independent, more extreme instance — full-corpus retrieval isn't merely less precise than
batch-of-32, it is not a coherent statistic in this embedding space at all.
**Does NOT establish:** anything about this project's own model, dataset, or findings — R-Precision
itself, always computed within the correct batch-of-32 protocol throughout this project
(E0a/E0b/E1A/E1B/E1-pilot), is unaffected by this finding. Whether other text-motion evaluators
(different training objectives, different embedding dimensions) share this same full-corpus
collapse — untested, a plausible but unverified generalization.
