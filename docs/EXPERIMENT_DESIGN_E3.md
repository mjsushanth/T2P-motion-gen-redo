# EXPERIMENT_DESIGN_E3 — does CLIP's spatial deficit actually reach generated motion?

**Status: RUN AND SCORED (2026-09-08).** Sections 1-8 below are the design as pre-registered,
unchanged from before the run. Results are in **section 9**.

## 1. The question

Notebooks 01 and 01b established that MDM's frozen CLIP text encoder under-separates spatial
language (left/right, forward/backward) from comparable non-spatial language, in embedding
space, on hand-written minimal pairs (n=40/group, Cohen's d=+0.995). **Whether that deficit
survives into the actual generated motion — on real HumanML3D captions, at the retrieval metric
this project's whole evaluation harness depends on — has never been measured.** E3 measures it
directly: generate every test-split caption once with the released checkpoint, no training, and
compare R-Precision on the spatial subset against the non-spatial subset.

## 2. Why this comes before E2

`docs/EXPERIMENT_DESIGN_E2.md` asks whether *repairing* CLIP's spatial encoding (a trained
adapter) improves generation on spatial captions specifically. That question is only
interpretable if E3 has already shown there is a real gap to close. If E2 ran first and returned
a null result, the null would be ambiguous on its face: did the adapter fail to help, or was
there never a generation-level gap for it to help with? **E3 removes that ambiguity, and it is
strictly cheaper and lower-risk than E2** — no training, no adapter, just generation and scoring
with infrastructure this project has already built and validated. Running it first is not
optional caution; it is what makes E2's own result interpretable either way it lands.

## 3. Design

**Generate:** every entry in HumanML3D's standard test-split evaluation pool, once each, with
the released checkpoint (`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/
model000475000.pt`), fixed seed, guidance scale 2.5, 1,000 diffusion steps (no respacing) — the
exact same generation convention every other number in this project has used (E0b, E1A, E1B). No
training, no new model, no adapter.

**A population correction made before running, not after:** the design originally assumed 4,198
samples, matching notebook 05's direct read of `test.txt` (one entry per unique sequence). The
actual evaluation-mode data loader — the same one E0b, E1A, and E1B already draw their own
R-Precision numbers from, just truncated to n=128 there — draws from a cached pool of **4,648**
entries: the same 4,198 base sequences plus 450 extra entries for sequences that have more than
one annotated caption (each extra entry letter-prefixed to stay a distinct dataset key, standard
HumanML3D convention). Restricting generation to an arbitrary 4,198-entry prefix of this pool
would be a new, one-off population no other number in this project uses. Generating the full,
standard 4,648-entry pool instead keeps this measurement on the exact same population every
other evaluation number here already stands on. **The spatial/non-spatial split is therefore
computed fresh from whichever captions actually get generated (§3 below), not assumed to
reproduce notebook 05's exact 2,448/1,750 counts** — those counts remain correct for their own
4,198-sequence population; this experiment's own realized counts are reported as measured.

**Score with both evaluators this project has working and cross-checked (notebook 02):** Guo's
R-Precision (euclidean ranking, its own published convention) and TMR's (cosine ranking, its
own). Batches of 32 for both, matching the field's own protocol.

**Report separately, not pooled:**
- the spatial subset (~2,700 captions expected, scaling notebook 05's 58.3% test-split rate to
  4,648 — the realized count is reported directly, not assumed)
- the non-spatial subset (~1,940 captions expected, same scaling)
- both subsets further split into caption-length terciles, computed once from each subset's own
  word-count distribution (same convention as `docs/EXPERIMENT_DESIGN_E2.md` §3) — because
  notebook 05 already found spatial captions run 41% longer on average (14.38 vs 10.19 words,
  Cohen's d=+0.592, not a large-n artifact), and any spatial-vs-non-spatial comparison that
  doesn't hold this out is not measuring what it claims to.
- FID per subset, as a secondary check (see §5).

## 4. Power — stated before any result exists

Using this project's own validated binomial-floor formula (`docs/LANDMINES.md` §18,
`n = z²·2p(1-p)/delta²`, p=0.35 matching this checkpoint's own severely-undertrained operating
range, z=3 for a 3σ threshold):

| n | MDE (3σ) |
|---:|---:|
| 128 (every prior generation run this project has done) | 0.179 |
| ~2,700 (expected spatial subset) | 0.039 |
| ~1,940 (expected non-spatial subset) | 0.046 |
| ~900 (one length tercile within the spatial subset, ~2,700/3) | 0.068 |

(Recomputed once the realized subset sizes are known, in the results themselves — these are
the pre-run estimates the design is committing to, not the final reported numbers.)

A **~4.4x** improvement in resolving power over every generation-side comparison this project
has run so far (E1A vs E1B included). Effects that were invisible at n=128 all session become
measurable at these subset sizes, without spending a single additional sample beyond what
generating the full test split already produces.

## 5. FID becomes trustworthy here for the first time

Notebook 04 measured FID's own real-vs-real estimator-bias floor directly: **1.577 at n=128**,
falling to **0.109 at n=2,000**. Every FID this project has computed so far (E0b, E1A, E1B) was
computed at n=128 and is dominated by that same bias, not by anything about model quality. At
n=1,750 and n=2,448, FID is close to or above the range notebook 04 showed is estimator-noise-free.
**This is the first FID this project reports at a sample size where the metric is not known to be
broken by construction** — reported per subset, cross-checked directly against notebook 04's own
floor curve rather than trusted on the strength of a bigger n alone.

## 6. Pre-registered interpretation — written before the run, not after

- **Spatial materially worse than non-spatial, beyond the MDE:** the embedding-space deficit
  propagates into generation. E2 becomes well-motivated, and this experiment states the actual
  gap size an adapter would need to close. **The caption-length confound makes this direction
  conservative, not inflated**: longer captions carry more retrievable signal (a more distinctive
  query is less confusable with other candidates in the same retrieval batch), and spatial
  captions are the longer group — so the confound should push the spatial subset's R-Precision
  *up*, working against this outcome. A deficit found despite that push is a real, understated
  one, not an artifact of the length difference.
- **No difference, or spatial better, beyond the MDE:** the embedding-space deficit does **not**
  propagate to generation at this checkpoint's operating range. This is the more surprising
  outcome of the two, and arguably the more valuable one: it would mean a text-encoder repair is
  aimed at a gap that doesn't reach the actual output, and it would save the cost of building and
  training an adapter for a problem that isn't there. **This outcome is reported with exactly the
  same prominence as the first**, not buried as a null result.
- **Effect inside the MDE at every subset size tested:** reported as indistinguishable from
  noise, exactly as this project's own prior stopping decisions (`docs/DECISIONS.md`
  D-24/D-27/D-28) already established it will do, rather than over-read a favorable- or
  unfavorable-looking number that the power calculation in §4 says this design cannot resolve.

## 7. Compute

**~4.07 hours** at this project's own directly measured local rate (3.15 s/sample, batch 32,
1,000 diffusion steps, `trans_enc` 17.9M-param architecture — the same rate `docs/
EXPERIMENT_DESIGN_E2.md` §7 derived and used), revised up from the original 3.67h estimate to
match the corrected 4,648-sample population (§3): 4,648 samples × 3.15 s/sample ≈ 14,641s ≈
4.07h. **This machine, not Kaggle** — same reasoning as E2 §7: this machine has already closed
every silent-failure surface on the MPS path that a fresh Kaggle environment would reopen, for a
hypothetical speedup nobody has measured there.

All 4,648 generated motions held in memory at once are small: (4648, 196, 263) float32 ≈ 0.96 GB
— comfortably resident, no disk round-trip needed during scoring. **Generation itself is
checkpointed incrementally to disk, one batch of 32 at a time**, independent of the in-memory
scoring step: an interrupted run resumes from the last completed batch rather than losing the
whole ~4-hour run, since the generation loop and the batch-caching are the same operation, not
an afterthought bolted on top. A 32-batch smoke test (n=64, two batches) was run first and timed
at 3.22 s/sample -- within noise of the 3.15 s/sample this estimate assumes.

## 8. What this does not claim

One checkpoint (this project's own severely-undertrained model, 3,000 of MDM's 475,000 published
steps), one seed, one dataset. This measures whether *this* model on *this* benchmark shows a
generation-level spatial gap — not a general property of text-to-motion systems, and not a claim
about what a fully-trained model would show. It is also not a claim that this checkpoint's
absolute quality is representative of the field (it isn't — see `RESULTS.md` §1.4); it is a
same-model, same-budget, within-checkpoint comparison, which is exactly what the spatial-vs-
non-spatial split needs and no more.

**Would reverse if:** the honest MDE at the subset sizes actually measured (§4) turns out larger
than any effect worth caring about once real variance is accounted for (e.g. if per-sample
retrieval outcomes turn out far more correlated within a batch than the independent-Bernoulli
assumption behind §4's formula assumes) — in which case this design's own numbers, not a
qualitative impression, are what say so.

---

## 9. Results

**Generated:** 4,640 samples (145 full batches of 32; the eval pool's own remainder of 8 was
dropped, same batch-of-32 convention used throughout this project), 4.03 hours wall-clock,
matching the ~4.07h estimate to within 1%. Split: **2,708 spatial (58.4%) / 1,932 non-spatial**
— consistent with notebook 05's own 58.3% corpus-wide rate to 0.1 point; the earlier concern that
E3's population might run more spatial than notebook 05's (from the 450 multi-caption entries)
did not hold up once the full run completed and is not carried forward.

### 9.1 R-Precision — the decisive comparison

| top-3 | spatial (n=2,708) | non-spatial (n=1,932) | gap | combined SE | σ | 3σ threshold |
|---|---:|---:|---:|---:|---:|---:|
| Guo | 0.5971 | 0.5901 | +0.0070 | 0.0146 | +0.48σ | 0.0439 |
| TMR | 0.6998 | 0.7260 | -0.0263 | 0.0134 | -1.95σ | 0.0403 |

**Neither evaluator clears the pre-registered 3σ threshold, and the two evaluators disagree even
on the *sign* of the (statistically insignificant) gap** — Guo reads spatial very slightly ahead,
TMR reads non-spatial ahead by a larger but still sub-threshold margin. Per §6's pre-registered
interpretation, this is the **"no difference, beyond the MDE" outcome**: the embedding-space
spatial deficit notebooks 01/01b measured (Cohen's d=+0.995 on CLIP's own embeddings) does **not**
show up as a resolvable generation-level R-Precision gap at this checkpoint's operating range,
under either evaluator, at this experiment's power (MDE 0.039 spatial / 0.046 non-spatial, a
~4.4x tighter floor than every prior generation-side comparison in this project). Reported with
the same prominence as a positive result would have received, per the design's own commitment.

Top-1/top-2 show the same pattern (largest single gap: TMR top-2, non-spatial ahead by 0.046,
still at the edge of that subset's own MDE, not clearing it).

### 9.2 Length terciles

Spatial terciles (mean caption length 7.4 / 12.5 / 23.0 words) show Guo top-3 rising with length
(0.509 → 0.587 → 0.598) while TMR top-3 is non-monotonic (0.623 → 0.699 → 0.622). Non-spatial
terciles (5.9 / 8.8 / 16.0 words) show the opposite pattern for TMR (falling: 0.738 → 0.753 →
0.678) and are roughly flat for Guo. Per-tercile n (~640-900) carries an MDE of ~0.07-0.08 —
comparable to or larger than every tercile-to-tercile swing observed. **No tercile pattern here
clears its own noise floor**; reported as a secondary, inconclusive observation, not a finding.

### 9.3 FID — secondary, with floors, per §5's own caution

| subset | n (gen / gt) | generated-vs-real FID | real-vs-real floor (n/side) | rough 1/n estimate |
|---|---|---:|---:|---:|
| overall | 4,640 / 4,640 | 0.484 | 0.096 (n=2,320) | 0.094 |
| spatial | 2,708 / 2,757 | 0.649 | 0.179 (n=1,378) | 0.159 |
| non-spatial | 1,932 / 1,883 | 0.437 | 0.257 (n=941) | 0.232 |

**Pipeline sanity check (per §5):** each subset's own real-vs-real floor lands within 2-13% of
the rough 1/n estimate at that same per-side n — consistent with a working pipeline, not a red
flag. (An earlier draft of this table divided the rough estimate by the full subset n rather than
the per-side n the floor is actually measured at, which made the two numbers look ~2x apart; that
was a bug in this notebook's own arithmetic, not a pipeline problem, caught and fixed before
being reported here.)

**Raw cross-subset FID comparison is invalid and not attempted**: spatial's generated-vs-real FID
(0.649) sits at 3.6x its own floor; non-spatial's (0.437) sits at 1.7x its own floor. Whether that
ratio-to-own-floor difference reflects anything real is not established by this design (FID is
secondary here, R-Precision is decisive) — noted as a secondary, exploratory observation only.

### 9.4 What this establishes

**The embedding-space spatial deficit measured in notebooks 01/01b does not propagate into a
resolvable generation-level R-Precision gap on this checkpoint, at this experiment's power.**
This is a genuine null, not an underpowered non-result: the design's own MDE (0.039-0.046) is
tight enough that a deficit anywhere near the embedding-space effect size would very likely have
shown up. Per §2's own reasoning, this bears directly on `docs/EXPERIMENT_DESIGN_E2.md`: a
text-encoder adapter aimed at closing a generation-level spatial gap would be repairing a defect
that, at this checkpoint's scale, does not measurably reach the output — E2 is not automatically
well-motivated by this result the way a positive finding here would have made it.

**What this does not rule out:** a larger, better-trained model might show a gap this severely
undertrained checkpoint (0.63% of MDM's published training budget) cannot resolve or does not
exhibit at all; the tercile analysis (§9.2) is underpowered to detect a length-interacting effect
even if one exists; and TMR's own -1.95σ reading, while not clearing the pre-registered threshold,
is the largest single number in this table and is not nothing -- a repeat at this same power on a
better-trained checkpoint would be the natural next check before concluding this generalizes.

### 9.5 A discrepancy against a smaller prior measurement, investigated directly

`notebooks/02`'s own n=128 measurement of this exact checkpoint, this exact evaluator, found Guo
R-Precision-top3 = 0.7578. E3's own n=4,640 measurement of the same quantity found 0.6172 --
using the n=128 estimate's own binomial SE, that is 3.71σ apart. Two measurements of the same
thing disagreeing this much is worth resolving before either number is trusted, not noted and
left.

**Hypothesis tested: batch-of-32 composition, not the model, explains the gap.** R-Precision
ranks each sample against 31 decoys drawn from its own batch; if consecutive samples in loader
order are more mutually similar than a random draw would be, decoys get easier and R-Precision
rises for that reason alone, independent of anything about the model. A real, measured
compositional drift motivated the test: the run's first 1,696 captions are 60.6% spatial, the
next 2,304 are 57.1% -- a genuine pattern in this cached generation, not asserted from theory.

**The test:** re-score the same, already-computed embeddings (no regeneration) with batch-of-32
assignment replaced by 20 independent random reshuffles of the full 4,640-sample pool, comparing
against the as-generated (natural loader) order.

| | top-3 |
|---|---:|
| natural order | 0.6172 |
| shuffled (mean of 20 trials, std 0.0048) | 0.6184 |

**Refuted.** Shuffling moves the estimate by 0.0012 -- about 1 SE of the shuffled mean, and 29
standard deviations short of the 0.7578 the hypothesis needed to explain. Batch composition is
not why E3's number differs from notebook 02's. **The n=128 measurement was a small-sample
outlier; E3's own n=4,640 figure, with a binomial SE roughly 5x tighter, supersedes it** as this
project's best estimate of this checkpoint's Guo R-Precision-top3.

**The spatial-vs-non-spatial null is robust to the same reshuffling**, exactly as predicted before
the test ran: shuffled spatial mean 0.6006, non-spatial mean 0.5847 (gap +0.0159, same sign and
similar magnitude as the natural-order gap of +0.0070). This gap-under-reshuffling is a
consistency check on the *sample*, not a new significance claim about the *model* -- the sample
itself, however batched, shows a small, same-direction gap; whether that gap reflects a real
population effect is still governed by §9.1's own combined-SE calculation (0.48σ), not by how
tightly reshuffling estimates the fixed sample's own gap. Both readings point the same way: no
resolvable spatial-vs-non-spatial effect, and that conclusion does not depend on how the 4,640
samples happen to be grouped into batches.

**A specific trap worth naming, because the number sits right next to the correct one.** The 20
shuffled trials permit a paired comparison of the spatial-minus-non-spatial gap across trials:
mean 0.0158, std 0.0086 across trials, giving a paired t=8.26 (p<0.001) -- verified directly, a
real, correctly-computable number. **Reporting that as evidence of significance would be wrong.**
All 20 trials share the same fixed 4,640 samples and differ only in batch assignment, so that
paired standard error captures batch-reassignment variance only, not the sampling variance of
which captions and generations this represents -- it answers "is this gap stable across
re-batchings of this exact sample" (yes, essentially by construction, since the underlying
2,708/1,932 split never changes), not "is this gap real in the population the sample was drawn
from." The correct uncertainty is the binomial one already used above (combined SE 0.0146, gap
0.0158 giving z≈1.08 using the shuffled-mean gap as the point estimate) -- comfortably inside
noise either way, but for the right reason. **General form, worth keeping for the next time this
pattern shows up:** resampling that permutes structure while holding the underlying sample fixed
measures robustness to that structure, never sampling uncertainty in the population the sample
represents -- a significance test built on such trials inflates apparent confidence, roughly by
the square root of the trial count.

**Consequence for `docs/DECISIONS.md` and future generation-side comparisons:** this project's
Guo R-Precision-top3 for this checkpoint should be cited as ~0.617 (n=4,640), not 0.7578 (n=128),
going forward. The 0.7578 figure was independently cross-checked to the digit against an earlier
measurement and treated as validated at the time -- it agreed with the *wrong* thing to agree
with, which is itself worth remembering: a cross-check against a small-sample number confirms
consistency between two samples, not that either one is close to the population value.

### 9.6 A second, larger result: this reproduces a published external number

E3's own comparison in §9.1 is internal -- spatial vs. non-spatial, both measured by this
project's own pipeline. Separately, and worth stating on its own: **E3's overall figure also
reproduces MDM's own published number.**

| | Guo R-Precision-top3 | n | SE (binomial) |
|---|---:|---:|---:|
| MDM, published (`LANDSCAPE.md` line 47) | 0.611 ± .007 | (paper's own protocol) | .007 (paper's own) |
| MDM, this project's harness (E3, overall) | 0.6172 | 4,640 | 0.0072 |

Difference: +0.0062, against a combined SE of ~0.0072-0.010 depending on which SE is taken as
authoritative -- **+0.62 to +0.87σ either way**, comfortably inside ordinary sampling noise. This
project's own evaluation pipeline, scoring the same released checkpoint the paper reports, on the
full 4,640-caption test-eval pool, lands on the paper's own number without having trained anything.
That is a genuine harness-validation result, independent of and in addition to the spatial null
in §9.1-9.4: it is evidence about *this project's measurement stack*, not about this checkpoint's
conditioning behavior. See `docs/DECISIONS.md` D-03 for what this changes about that gate's status.

**A related question, worth closing out precisely rather than loosely: was the superseded n=128
figure (0.7578, §9.5) ever a plausible reading of the *published* 0.611, on its own terms?**
Comparing 0.7578 against 0.611 needs the standard error of *the measurement that produced 0.7578*
-- i.e. the SE at n=128, not the tighter SE E3 achieved at n=4,640. Using the correct, matching SE:

```
SE at n=128, using the published p=0.611:  sqrt(0.611 * 0.389 / 128) ≈ 0.0431
z = (0.7578 - 0.611) / 0.0431 ≈ 3.41σ
```

That is a real discrepancy (roughly a 1-in-1,500 two-tailed tail event) -- consistent with §9.5's
own conclusion that 0.7578 was a small-sample outlier, not a trustworthy figure for this checkpoint.
**It is not 20+ sigma.** Reusing E3's own tighter n=4,640 SE (≈0.0072) against the n=128 figure's
discrepancy gives (0.7578-0.611)/0.0072 ≈ 20.5 -- a number that looks far more dramatic and is
*wrong*, for the same reason this project's own FID "rough floor" sanity check briefly looked
broken earlier in this document's history (§9.3): dividing a measurement's own discrepancy by a
different measurement's tighter uncertainty always inflates the apparent significance. The
correct, matching-SE answer is ~3.4σ. Both readings support the same conclusion -- 0.7578 was not
a good estimate of this checkpoint -- but only one of them is an admissible statistic, and it is
worth being precise about which, in a document that exists specifically to catch this class of
mistake.

### 9.7 A comparison against the published FID -- with a bootstrap error bar, and explicit limits

§9.3's overall generated-vs-real FID (0.484) was never put beside MDM's published FID (0.544,
CI half-width 0.044 over 20 replications, `LANDSCAPE.md` line 47) anywhere in this document --
an oversight caught the same way the R-Precision reproduction was initially missed (§9.6): two
real numbers, in two different files, nobody had opened side by side. Doing so now, with the
three caveats this comparison actually needs stated up front rather than implied.

**The comparison needs an error bar first.** A single FID computed from one fixed generation and
one fixed reference pool has no variance information on its own -- it cannot be judged "close to"
or "far from" a 20-replication published figure without one. `scripts/e3_fid_bootstrap.py`
re-embeds the same cached generated motions and the same ground-truth pool (no regeneration),
then resamples both pools independently, with replacement, at their own n=4,640, 1,000 times,
recomputing FID each trial:

| | value |
|---|---:|
| point estimate (this script's own re-embedding) | 0.4858 |
| point estimate (originally reported, §9.3) | 0.4838 |
| bootstrap mean (1,000 trials) | 0.5375 |
| bootstrap std | 0.0755 |
| bootstrap 95% CI | [0.403, 0.698] |
| published (paper's own 20-replication figure) | 0.544 ± .044 |

The two point estimates differ by 0.0020 (~0.4% relative) despite being computed from identical
cached motions -- a small residual, most likely floating-point ordering sensitivity in the
covariance/Frechet-distance computation given the two scripts embed in slightly different code
paths, not investigated further since it is far smaller than anything below.

**The bootstrap mean is not the estimate to use, and is not used here.** Resampling with
replacement duplicates motions and shrinks effective diversity, biasing FID upward by
construction -- this run's bootstrap mean (0.5375) sits 0.0517 above its own point estimate
(0.4858), which is that bias, not a second measurement. The point estimate (0.4858) is the
number compared against published; the bootstrap std (0.0755) is used only as a dispersion
estimate, not the bootstrap mean as a center.

**Sizing the claim precisely, using the point estimate plus/minus twice the bootstrap std as the
accepted interval:** [0.335, 0.637]. The published figure (0.544) falls inside it, comfortably.
**The correct statement is "not distinguishable from the published figure under an uncertainty
this test cannot tighten" -- not "reproduced."** That distinction matters because the interval is
wide: +/-15.5% around the point estimate, against R-Precision's +/-1.2% (0.6172 +/- 0.0071,
§9.6). The two halves of this gate are not the same kind of evidence:

| | value +/- uncertainty | relative window | verdict |
|---|---|---|---|
| R-Precision (§9.6) | 0.6172 ± 0.0071 | ±1.2% | tight match |
| FID (this section) | 0.4858 ± 0.0755 | ±15.5% | non-rejection, not a tight match |

A test with a ±15.5% window has correspondingly low power: E3's FID would have read as "not
distinguishable from published" for almost any true value the paper might have reported in
roughly [0.335, 0.637], and the true window is wider still once genuine between-replication
variance (not available from this bootstrap, see point 1 below) is admitted. Worth having, and
not the same strength of claim as the R-Precision result -- lead with R-Precision wherever both
are cited together.

**What this is not, stated as plainly as the result itself.** Three things this bootstrap does
NOT establish:

1. **This is not the paper's own replication variance.** The paper's ±.044 comes from 20
   independent generation runs (different sampled noise, different diffusion trajectories) and
   a fixed reference protocol. This project's bootstrap resamples *the same fixed 4,640
   generated motions and the same fixed 4,640 real motions* -- it measures how much the FID
   estimate itself wobbles when its own two fixed pools are resampled, not how much a genuinely
   independent regeneration would move it. These are related but different quantities; the
   bootstrap CI is a lower bound on the true between-replication spread, not a substitute for it.
2. **The reference protocol may differ.** FID is acutely sensitive to which real motions form
   the reference set and how many (`docs/LANDMINES.md` §14). This project's ground-truth pool
   (`embed_ground_truth_guo`, the 4,648-entry eval-mode pool) was not verified to match MDM's own
   published-run reference construction sample-for-sample -- only shown consistent with this
   project's own pre-existing `fixed_gt_reference.npz` cache (§0 setup, prior work). A protocol
   mismatch, not model quality, could account for some or all of the 0.058 gap.
3. **The direction is not evidence of a better model.** This project's point estimate (0.484-
   0.486) sits *below* the published 0.544 -- nominally "better." Given a bootstrap std of 0.076,
   being 0.058 below published is well inside one standard deviation and carries no directional
   signal: it is equally consistent with a true match, a genuinely better result, or a protocol
   difference that happens to read low. Reading it as "this project beat the published number" is
   not supported by this evidence and is not claimed here.

**What would actually tighten this, and what it would cost.** The bootstrap above cannot supply
what is missing (point 1) -- only independent regenerations, with different sampled noise, can
give a real between-replication spread of the kind the paper's own ±0.044 measures. At the
measured **4.03h wall-clock on MPS** per full-scale replication (`artifacts/e3/generate.log`),
**three total replications cost ~12.1h of MPS compute** (one already exists; ~8.1h of
additional compute for two more independent runs) and would yield a genuine, small
between-replication spread -- a real tightening of the FID half, not another bootstrap. A full
20-replication protocol matching the paper's own would need 19 more runs beyond the one that
exists -- **19 x 4.03h ~= 76.6h of additional MPS compute** (~80.6h total across all 20, if
counted from scratch). Both figures are recorded here as the priced
cost of closing this gate, not as work undertaken -- whether 3 replications (a cheap, honest
partial tightening) or the full 20 is worth spending against other candidate work is the
author's own call, not this session's; noted as the next candidate experiment for this gate,
not launched. See `docs/DECISIONS.md` D-03 for the updated statement.

**Net reading, stated as a verdict rather than left open-ended:** **non-rejection achieved,
tight reproduction not achieved.** This bootstrap rules out "the gap is huge and obviously a
broken pipeline" -- the published figure sits inside a reasonable, if wide, uncertainty band for
a single measurement at this n. It does not establish a reproduction to any tight tolerance, for
the three reasons named above, chief among them that the interval able to say so is ±15.5%, not
R-Precision's ±1.2%. Recorded as suggestive-but-inconclusive, not as a second reproduction
alongside §9.6's R-Precision result -- and not to be cited later as "E3 reproduced the published
FID" without this section's own caveats attached.
