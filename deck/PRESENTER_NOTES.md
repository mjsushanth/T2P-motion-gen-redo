# PRESENTER_NOTES — provenance for every number on a face

Planner-owned. C1 pastes these into the `<!-- -->` block of the matching slide in `slides.md`.

**Rule this file enforces:** every figure on a slide face traces to a source opened in the
session that wrote it, cited to file and line. Not remembered, not relayed. Where a number
could not be sourced, the claim was rewritten so it did not need one — see slide 2.

---

## Slide 2 — predecessor architecture

No number on this face, deliberately. An earlier draft carried "23.3K training samples," taken
from the author's own account of the earlier project rather than from this repository. A grep
across every markdown file here returns no such figure, so it was removed rather than softened.
The four resolution levels appear in the diagram as structure, which the diagram may depict as a
design fact without asserting a measurement.

No "failed", "broken" or "wrong" appears on this face or in this diagram. That is a standing
editorial rule for the whole deck, recorded in `DECK_PLAN.md` §2.

## Slide 3 — instrumentation

**17 pipeline and metric failure modes** — `docs/LANDMINES.md`, §1-15 plus §24-25. Corrected down
from 25 during review: the document has 25 numbered sections, of which 8 are review-discipline
lessons about the practice of producing research rather than traps in the pipeline or metric stack.
**Precision note:** the document's own heading at `:820` reads "Review-discipline lessons (§16-21)"
and covers six of those eight. §22 and §23 are classified the same way by reading them, not by the
document's own label. The count of 17 holds either way; the justification is part citation and part
judgement, and should be stated that way rather than attributed wholly to the heading.

**25.81% mean bone-length CV, up to 81.70%** — `FORENSICS.md:102` (mean across bones) and
`FORENSICS.md:93` (the 81.70% bone, joints 12-15). The decision rule was stated before the
measurement ran: `FORENSICS.md:105`.

## Slide 5 — reproduction

**0.6172 measured against 0.611 ± 0.007 published, n = 4,640** —
`artifacts/e3/e3_record.json`, `published_reproduction` block.

**The face deliberately states the gap, not a sigma, and here is why.** The project's canonical
0.87σ (`e3_record.json` `z_sigma`, repeated in `docs/DECISIONS.md` D-03) divides the 0.0062
difference by the measured binomial SE alone (0.0062/0.0072 = 0.867), a one-sample z treating the
published value as a fixed constant. But the published value carries its own stated ±0.007, so the
conventional two-sample statistic pools both in quadrature: 0.0062/√(0.0072²+0.007²) = **0.62σ**.
Both are defensible; they answer different questions. Rather than defend a convention in front of
an audience that may check the arithmetic, the face states the quantity that needs no convention:
the difference is 0.0062 and the reference's own error bar is ±0.007, so the gap is smaller than
the uncertainty of the thing being reproduced.

**If asked for a sigma, say 0.62σ and explain why**, not 0.87σ. The quadrature figure is the
conventional one and it is the *more* favourable of the two — it shows more overlap, not less.
Caught during figure review by the worker building fig05, who re-derived it rather than
transcribing it. The project's own records still say 0.87σ; that is a convention difference, not
an error in them, and it is not worth propagating a correction through D-03 for a number no slide
now uses. Cross-referenced at
`docs/DECISIONS.md` D-03, which records this as the decisive form of the gate because it
validates checkpoint loading, generation and scoring end to end.

**The FID half is a non-rejection, not a tight reproduction** — `docs/DECISIONS.md` D-03. The
bootstrap interval is ±15.5% against R-Precision's ±1.2%, and it resamples one fixed generation,
so it measures resampling stability rather than the paper's own between-replication variance.
Not spoken as a reproduction under any circumstances.

**Ground-truth support cites E0b (0.7969), never E0a (0.7202).** E0a's script hand-builds its own
config and loader; that reconstruction carried a bug E0b's does not. Both values are still live
in the repo. Do not quote E0a.

## Slide 6 — encoder probe

**0.9707 spatial vs 0.9442 non-spatial modifier, Cohen's d = +0.995** — executed output of
`notebooks/01_clip_spatial_blindness.ipynb`. Mann-Whitney one-sided p = 0.00000, clears the
Bonferroni threshold of 0.0167 for the three comparisons run.

**0.72 for two unrelated topics** — same notebook's calibration cell. This is on the face because
without it a reader cannot tell whether 0.97 is high.

**Power fixed before the data** — n = 40/group, achieved power 0.913; the pre-registered
requirement was 27.6/group for 80%. Pilot (n=16, d=+0.678, p=0.034) and extension (n=24,
d=+1.266, p=0.00001) each independently significant.

**58.4% of captions contain spatial vocabulary** — `docs/EXPERIMENT_DESIGN_E3.md:156`
(2,708 of 4,640), corroborated independently by notebook 05's corpus-wide 58.3%.

**If asked about the pairs being hand-written:** concede it. The sample size was fixed by a
power calculation before the data and the two halves replicate independently, but the pairs
were authored by the same person testing them. The 58.4% figure shows the phenomenon is
common in the benchmark; it does not show these particular sentences are representative of
it. That is a live limitation and conceding it costs nothing.

## Slide 7 — pooling probe

**Three representations** — executed output of `notebooks/01b_pooling_probe.ipynb`: pooled gap
+0.0265 / d +0.995; mean-over-tokens +0.0185 / d +0.589; most-divergent-token +0.1106 / d +0.724.
Bonferroni alpha 0.0167 for three comparisons.

An earlier draft of the plan said the gap "widens" once pooling is removed. **It does not widen
monotonically** — the raw gap is largest at the token level while the standardised effect is
largest for pooled, and mean-over-tokens is the weakest of the three. The face says survival, not
growth. Do not restore the earlier wording.

## Slide 8 — propagation test

**+0.0070, SE 0.0146, +0.48σ against a pre-registered 3σ threshold of 0.0439** —
`docs/EXPERIMENT_DESIGN_E3.md:163-166`, the project's own §9.1 table. The planner re-derived these
independently from `artifacts/e3/e3_record.json` under both pooled and unpooled two-proportion SE
(+0.477 and +0.476) before finding the table; the two agree to the digit.

**MDE 0.039 spatial / 0.046 non-spatial, ~4.4x tighter than every prior generation-side
comparison here** — same section. This is why the null is worth a slide: the test had the power
to have found something.

**If asked whether the null just means the model has no headroom to lose:** overall top-3 is
0.617 against a ground-truth ceiling near 0.797 on the same instrument — roughly 0.18 of
headroom. A spatial deficit had room to appear and did not. This is the most likely live
question on this slide.

**If asked about the split being lexical:** concede it. A caption can contain a directional
word incidentally, and can require spatial reasoning without using one. The tercile control
handles caption length, not this. It is the cleanest thing a reviewer could ask to see
improved.

**Denominator note, not on the face.** Subset sizes are 2,708 and 1,932, but R-Precision scores
only complete 32-candidate batches, so the effective denominators are 2,688 and 1,920 — confirmed
by recovering exact integer success counts (1,605/2,688 Guo spatial, 1,133/1,920 Guo non-spatial).
The project's own table quotes the subset sizes. The difference moves the SE by less than 0.0001
and changes no conclusion, so the face carries no n at all rather than carrying a contested one.

## Slide 9 — second evaluator

**−0.0263, SE 0.0134, −1.95σ, threshold 0.0403** — `docs/EXPERIMENT_DESIGN_E3.md:166`.

**r = 0.404 per-sample correlation, n = 4,640** — `artifacts/e3/evaluator_correlation.json`,
computed by `scripts/e3_evaluator_correlation.py` reusing `e3_score.py`'s validated embedding
conventions with no regeneration. r² = 0.163, so the two evaluators share about 16% of their
per-sample variance and 84% is not shared. That is what the face means by "overlapping but not
identical."

**Do NOT claim the two halves differ in agreement.** The record also reports spatial r = 0.392
and non-spatial r = 0.421. Under a Fisher z test on the effective denominators that difference is
z = −1.17, nowhere near significant. It is in the record; it is not a finding.

**The withdrawn figure flattered us, which is why it needed catching.** An earlier draft cited
r = 0.328
from `notebooks/02_tmr_second_evaluator.ipynb` and `README.md:60`. That value was computed on the
**n = 128** cached E0b generation — the same sample whose Guo R-Precision-top3 of 0.7578 was later
shown to sit **3.71σ** from the n = 4,640 value of 0.6172 (`docs/EXPERIMENT_DESIGN_E3.md:231-232`),
and which that document explicitly rules out for forward citation (`:286`). Using a statistic from
that sample to license a conclusion about the n = 4,640 experiment pairs two different populations.
Caught by the implementer during source verification, not by the planner who wrote it.

The full-scale value came back **higher** — 0.328 to 0.404, r² from 10.8% to 16.3%. The withdrawn
number made the two evaluators look *less* alike than they are, which made the sign flip easier to
attribute to instrument-dependence than the evidence supports. The error ran in the direction of
our own argument. That is the kind a motivated check does not find, and it is the reason the rule
is to verify the sample rather than to verify the conclusion.

**"Both are reported with the prominence a positive result would have had"** is not a flourish —
it is the design's own pre-registered commitment, `docs/EXPERIMENT_DESIGN_E3.md` §9.1.

**Do not claim either evaluator is correct.** If challenged that a disagreement means one is
broken rather than that the result is instrument-dependent, the r = 0.328 correlation is what
licenses the weaker and defensible reading: the two measure overlapping but non-identical
things. The face makes no claim about which is right, and should not start.

## Slide 10 — length confound

**14.31 vs 10.22 mean caption words** — `artifacts/e3/e3_record.json`, per-subset
`mean_caption_words`.

**Guo terciles: spatial 0.509 → 0.587 → 0.598; non-spatial 0.572 / 0.544 / 0.569** —
`artifacts/e3/e3_record.json` tercile blocks, narrated at `docs/EXPERIMENT_DESIGN_E3.md:183-185`.

**Scoped to the primary evaluator on purpose.** On TMR the same spatial terciles are
non-monotonic (0.623 → 0.699 → 0.622) and the non-spatial ones fall. The pattern is a property of
one instrument and the face says so. An unscoped version of this claim would be false.

## Slide 11 — FID floor

**Real-vs-real floors: 0.0957 (n=2,320/side), 0.1791 (n=1,378), 0.2573 (n=941), 30 trials each** —
`artifacts/e3/e3_record.json`, `fid_results.*.real_vs_real_floor`.

**1.0731 / 1.3997 / 3.2909 on one bit-identical set of 128 motions** — `docs/LANDMINES.md:487-489`.

**131,328 covariance parameters** — a 512x512 symmetric covariance has 512x513/2 unique entries.
Arithmetic, stated as arithmetic.

**No fitted curve on the figure, binding.** This project previously claimed a `C/n` floor law fit
"to 0.5%" from two points; 50 trials later found a spread of roughly 20%. Points and error bars
only. The record of that correction is why the rule exists.

## Slide 12 — bounds

**Effects ≥0.175 excluded at 3σ, ≥0.117 at 2σ** — `docs/DECISIONS.md:103-104`, D-24/27/28.

**The declined experiment** — same decision record. Once the MPS path was fixed the comparison
cost about an overnight run rather than being unaffordable, and it was closed anyway because the
0.0469 target was the experiment's own seed-noise reading at 0.80σ, not a hypothesis-motivated
effect size. The honest framing is a decision, not a limitation.

## Slide 13 — next

**4.03h per full replication; ~8.1h for three; ~76.6h for the published twenty** —
`docs/DECISIONS.md:61,63,65` (D-03). These supersede `RESULTS.md`'s pre-MPS CPU figures
("~5 CPU-hours", "roughly 100"), which are stale and must not be quoted.

**arXiv:2609.05730, *Bigger Text Encoders Can Hurt CLIP Zero-Shot Performance*,** Char,
Domingo-Enrich, Balestriero, submitted 4 Sep 2026. Abstract and released analysis data fetched
directly this session from arxiv.org and github.com/samirchar/clip-asymmetry (MIT). 5 vision x 6
text encoders = 30 models.

**The face says "reports", not "finds".** Their own abstract hedges to "for most vision encoders,"
and their released `degradation.json` shows the effect significant for the two smaller vision
encoders (p = 0.0014, p = 0.0003) and not for the two larger (p = 0.099, p = 0.585). Attribute it
as their result, stated at their own strength, and do not upgrade it.

---

## Numbers barred from every face

- `RESULTS.md` generation costs — superseded by D-03 (see slide 13).
- `RESULTS.md` §2's claim that the published number could not be reproduced at any affordable
  sample size — superseded by E3.
- E0a's 0.7202 ground-truth figure — script defect, cite E0b's 0.7969.
- The MPS training speedup. `docs/DECISIONS.md` D-24 says 9.95x; an independent measurement gave
  9.89x. Unresolved, not load-bearing for any slide, and therefore on no slide.
- **Anything computed on the n=128 cached E0b generation.** That sample's Guo R-Precision-top3
  (0.7578) is 3.71σ from the full-scale value and the project has ruled it out for forward
  citation. This bars the 0.328 evaluator correlation, the n=128 R-Precision values, and any
  derived statistic from that generation. If a number's provenance is that cache, it does not go
  on a face regardless of how well it was computed.
