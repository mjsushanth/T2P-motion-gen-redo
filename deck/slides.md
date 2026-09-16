---
theme: seriph
background: null
class: text-left
colorSchema: light
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
title: T2P-Reboot — Taking a text-to-motion pipeline apart
---

# T2P-Reboot
## Taking a text-to-motion pipeline apart

One released checkpoint &middot; 4,640 generated sequences &middot; one laptop

<!--
DECK STATUS: 12 of 13 slides built and verified against source (see per-slide notes below for
exact citations). Restructured 2026-09-16 (Joel's direct instruction): every figure-bearing
slide now uses the two-column layout (bullets left ~65%, figure right ~35%) instead of
stacking the image below the text -- the earlier single-column layout forced the image to
compete with the whole text stack for vertical room, which produced the tiny per-slide caps
this file no longer has. Only slide 2's architecture diagram remains -- implementer-authored,
held for last since it is the one figure with no measured number behind it and needs its own
review before landing (planner wants to see it before it's committed).
-->

---
layout: two-cols
layoutClass: wide-left
---

# The predecessor system: conditioned diffusion over 3D human pose

Before measuring a pipeline, you have to see what it was built to do.

- A UNet diffusion backbone conditioned on CLIP text embeddings, injected by cross-attention at four resolution levels — coarse layers carrying global posture, finer layers limb and hand placement.
- Anatomical plausibility enforced during training: bone-length consistency, joint limits, and kinematic-chain validation from pelvis through the extremities.
- Classifier-free guidance trading diversity against adherence to the prompt, ablated over guidance strength and noise schedule.

These are the right things to want from a text-to-pose model. The open question is how you would know whether you got them.

::right::

<img src="/img/architecture.svg" alt="Left-to-right pipeline: caption text into a frozen CLIP text encoder, projected into pose-conditioning space, injected via cross-attention into a four-resolution-level UNet diffusion backbone, producing 66-dimensional skeletal pose output, with bone-length, joint-limit and kinematic-chain losses annotated at training time">

<!--
deck/PRESENTER_NOTES.md: "No number on this face, deliberately. An earlier draft carried
'23.3K training samples,' taken from the author's own account of the earlier project rather
than from this repository. A grep across every markdown file here returns no such figure, so
it was removed rather than softened. The four resolution levels appear in the diagram as
structure, which the diagram may depict as a design fact without asserting a measurement. No
'failed', 'broken' or 'wrong' appears on this face or in this diagram -- standing editorial
rule, DECK_PLAN.md S2."

Architecture diagram: planner-authored (deck/img/architecture.svg, 400x660 portrait, teal
accent, "FROZEN -- NEVER UPDATED" label on the text encoder -- load-bearing for slides 6/7).
CFG deliberately not drawn (see PRESENTER_NOTES.md slide 2 for the reasoning: drawing it at
sampling time implies a corrected architecture, drawing it in the loss puts a defect on a
neutral-establishment slide; the bullet already covers it as a conditioning-strength control).
-->

---
layout: two-cols
layoutClass: wide-left
---

# Measurement: what the pipeline reports once it is instrumented

A model that produces plausible output and raises no exception cannot be judged by looking at it.

<div class="eyebrow">Pipeline and metric failure modes, verified not asserted</div>
<span class="stat">17<span class="stat-label">that change the answer and raise no error</span></span>

- Rebuilding with measurement at every stage produced a catalogue of silent failure modes in the pipeline and the metric stack, each verified rather than asserted.
- The load-bearing one is a decode check. Bone lengths in this dataset are constant by construction; measured under the original slice they vary by 25.81% on average, and up to 81.70% on individual bones.
- The number is not the finding. The invariant is — bone length is checkable with no model, no label and no metric.

::right::

<img src="/img/03_skeleton_comparison_stacked.png" alt="Two frames, well separated in time: the original decode stays an incoherent tangle at both, the dataset's own decode is a recognizable human figure whose pose visibly changes between them">

<!--
deck/PRESENTER_NOTES.md: "17 pipeline and metric failure modes -- docs/LANDMINES.md, S1-15
plus S24-25. Corrected down from 25 during review: the document has 25 numbered sections, but
S16-S23 are review-discipline lessons about the practice of producing research ... not traps in
the pipeline or the metric stack." Implementer verification (re-checked 2026-09-16): the
document's own 'Review-discipline lessons' heading (LANDMINES.md:820) literally scopes itself
to S16-21, not S16-23 -- S22/S23 substantively read the same way (process/review-discipline,
not pipeline/metric) but are the planner's own classification, not the heading's. The count of
17 (S1-15 + S24-25) still holds; planner has updated PRESENTER_NOTES.md to state this as
part-citation, part-judgement rather than attribute it wholly to the heading.

25.81% mean bone-length CV, up to 81.70% -- FORENSICS.md:102 (mean across bones) and
FORENSICS.md:93 (81.70%, bone 12-15). Both re-verified directly against the file, exact match.

FIGURE: replaced 2026-09-16. The original notebooks/03_skeleton_side_by_side.png (a 2-row x
4-column strip, aspect 0.539) was the one image in the deck the figure worker never
produced -- a pre-existing notebook output copied in, so both figure-legibility passes
scoped to "figures the worker made" missed it. Below the 240px floor at this deck's
35%-column width (peer-verified ~193px predicted, ~171px measured on the live deck at an
emulated viewport). Regenerated as a column-friendly, near-square comparison
(deck/img/03_skeleton_comparison_stacked.png, deck/img/make_fig03_skeleton_stacked.py,
0.925 aspect after a label-wording fix below, measured directly from the saved file after a
tight-bbox-crop iteration, not assumed from the requested figsize) -- 2 rows (frames, the
widest available temporal separation from the original 4-point strip, not reduced to 1) x 2
columns (the original decode | the dataset's own decode), preserving the "incoherent over
TIME, not a single bad pose" argument rather than the peer's literal single-frame
suggestion. notebooks/03_skeleton_side_by_side.png itself and its generating notebook cell
(03_263d_representation_and_f1_bug.ipynb, cell af057b27) are untouched -- this is a
separate, deck-only asset, not an edit to the notebook's own primary record.

Labels fixed 2026-09-16, caught by the planner against two rules already in force for this
deck: the panels originally read "wrong slice [:66]" (violates DECK_PLAN.md S2's binding
no-"wrong"/"failed"/"broken" rule about the predecessor) and the title carried the source
filename (sample004077.npy) with "recover_from_ric" as a panel label (violates house-style
rule 3: no internal filenames or function names on a face). Relabelled to "the original
decode" / "the dataset's own decode" and a filename-free title -- same argument, same
geometry, nothing evaluative or internal left on the face. Provenance, for the record
rather than the face: sample004077.npy, frames 38 and 130 of that sequence; left panel is
motion[:, :66] reshaped naively (the predecessor's own method); right panel is
recover_from_ric from third_party/motion-diffusion-model's own motion_process.py, unmodified.
-->

---

# The question: what does a benchmark score certify?

A text-to-motion model is scored by a retrieval benchmark. The published figure for the reference model is **0.611**.

That number cannot, on its own, separate a model that followed the sentence from one that produced a plausible motion.

Everything that follows is an attempt to find out which.

<!--
deck/PRESENTER_NOTES.md cites e3_record.json published_reproduction.published_value for 0.611;
also LANDSCAPE.md line 47 (paper's own table), both already independently verified earlier
this project. No figure, argued exception per DECK_PLAN S5's own framing.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Measurement: does this harness reproduce a published number?

An instrument that has never agreed with an external reference is not yet an instrument.

<div class="eyebrow">The gap, against the published figure's own uncertainty</div>
<span class="stat">0.0062<span class="stat-label">vs the published figure's own &plusmn;0.007, n = 4,640</span></span>

- Generated the full test split from the released checkpoint and scored it: 0.6172 against the paper's own 0.611 ± 0.007. The difference is smaller than the reference's own stated uncertainty.
- This validates the whole path end to end — checkpoint loading, generation, scoring — not only the evaluator's ability to re-encode motion that was already correct.
- The FID half of the same check is a non-rejection, not a tight reproduction. That distinction is kept open rather than counted as met.

::right::

<img src="/img/fig05_reproduction.svg" alt="Published and measured R-Precision values as overlapping intervals with plus-or-minus one standard error labelled, showing the gap smaller than the published figure's own uncertainty">

<!--
deck/PRESENTER_NOTES.md (updated): "0.6172 measured against 0.611 +- 0.007 published, n =
4,640 -- artifacts/e3/e3_record.json, published_reproduction block." Verified against
e3_record.json and DECISIONS.md earlier this project (matches exactly).

Stat reworded per planner's re-derivation, independently checked: the project's canonical
0.87 sigma (e3_record.json z_sigma, docs/DECISIONS.md D-03) is a one-sample z (0.0062/0.0072
= 0.867, treating the published value as a fixed constant). The published figure carries its
own +-0.007, so the conventional two-sample quadrature statistic is 0.0062/sqrt(0.0072^2+
0.007^2) = 0.62 sigma -- re-verified directly: sqrt(0.0072^2+0.007^2)=0.01004,
0.0062/0.01004=0.6175. Both are correct, different questions; the face now states the
convention-free quantity (the gap vs the reference's own error bar) instead of picking a side.
Note: quadrature (0.62 sigma) is the MORE favorable reading, not less -- the project's own
0.87 sigma was the more conservative of the two, not an overclaim.

This is a convention difference, not an error in docs/DECISIONS.md D-03 or
artifacts/e3/e3_record.json (both still say 0.87 sigma, correctly, under the one-sample
convention they state) -- not reconciling those records for this alone, since no other slide
depends on the discrepancy and both numbers remain individually defensible. Worth a light
project-record note about the convention difference at some point; not urgent.

"The FID half is a non-rejection, not a tight reproduction ... the accepted interval is
+-15.5% against R-Precision's +-1.2%." Independently verified (docs/DECISIONS.md D-03,
docs/EXPERIMENT_DESIGN_E3.md S9.7). "Ground-truth support cites E0b (0.7969), never E0a
(0.7202)." Verified.

FIGURE: fig05_reproduction.svg regenerated with the convention-free annotation and explicit
+-1 SE labelling; planner confirmed by opening it directly (not just the worker's report).
Unblocked 2026-09-16.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Experiment: can the text encoder tell "left" from "right"?

If the conditioning signal never carries direction, nothing downstream can recover it.

<div class="eyebrow">Spatial against comparable non-spatial contrasts</div>
<span class="stat">d = 0.995<span class="stat-label">Cohen's d, both directions clear Bonferroni</span></span>

- Sentences differing only by *left* / *right* sit at 0.9707 cosine similarity. Sentences differing by a comparable non-spatial modifier, in the same syntactic slot, sit at 0.9442.
- Scale first: two sentences on entirely unrelated topics still score 0.72. This encoder compresses ordinary English into a narrow band near the top, so the question is never "is 0.97 high" — it is "is 0.97 higher than it should be."
- Sample size was fixed by a power calculation run before the full data. The pilot and the extension are each independently significant, which forecloses the objection that easier pairs were written the second time.
- 58.4% of the benchmark's captions contain spatial vocabulary.

::right::

<img src="/img/fig06_clip_separation.svg" alt="Paired strip plot of cosine similarity for spatial versus non-spatial minimal-pair contrasts, 40 pairs per group">

<!--
deck/PRESENTER_NOTES.md: 0.9707/0.9442, Cohen's d=+0.995 -- executed output of
notebooks/01_clip_spatial_blindness.ipynb, Mann-Whitney one-sided p=0.00000, clears
Bonferroni 0.0167 for three comparisons. 0.72 unrelated-topics calibration same notebook.
Power: n=40/group, achieved power 0.913 (pre-registered requirement 27.6/group for 80%);
pilot (n=16, d=+0.678, p=0.034) and extension (n=24, d=+1.266, p=0.00001) each independently
significant. 58.4% -- docs/EXPERIMENT_DESIGN_E3.md:156 (2,708/4,640), corroborated by
notebook 05's 58.3% corpus-wide. These figures were already independently verified earlier
this project (notebook 01/01b executed outputs); not re-run fresh this session, but consistent
with the project's own prior record throughout.

If asked about the pairs being hand-written: concede it (per PRESENTER_NOTES.md) -- power
calculation and independent replication of the two halves stand, but the pairs were authored
by the same person testing them. 58.4% shows the phenomenon is common in the corpus; it does
not show these specific sentences are representative. Live limitation, costs nothing to concede.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Experiment: is the deficit made by pooling, or already inside the encoder?

The cheap explanation has to be eliminated before the expensive one is worth testing.

<div class="eyebrow">Representations tested; the gap survives in all of them</div>
<span class="stat">3 of 3<span class="stat-label">pooled, mean-over-tokens, most-divergent-token</span></span>

- The model reads one pooled vector per caption — that vector is the entire conditioning signal, so its geometry is exactly what the generator sees. If pooling were discarding the distinction, the per-token features would still carry it.
- They do not. Pooled, mean-over-tokens, and most-divergent-token each keep a significant gap after Bonferroni correction for the three comparisons.
- The orderings differ — the raw gap is largest at the token level, the standardised effect largest for pooled — so the claim is survival, not growth.
- This matters because it relocates the weakness onto a frozen component. Nothing in this pipeline updates the text encoder, so more motion data will not move it.

::right::

<img src="/img/fig07_pooling.svg" alt="Bar comparison of the spatial-vs-non-spatial gap across three CLIP representations: pooled, mean-over-tokens, most-divergent-token">

<!--
deck/PRESENTER_NOTES.md cites notebooks/01b_pooling_probe.ipynb executed output for all three
representation numbers (pooled +0.0265/d+0.995, mean-token +0.0185/d+0.589, most-divergent
+0.1106/d+0.724), consistent with this project's own prior, separately-verified record of the
same notebook. Planner correction: earlier draft said the gap "widens" -- it does not
monotonically; the face says survival, not growth, per instruction not to restore the earlier
wording.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Experiment: does the encoder deficit reach the generated motion?

A weakness in a component is not automatically a weakness in the system.

<div class="eyebrow">Measured difference, beside its own standard error</div>
<span class="stat" style="font-size:1.3em">+0.0070 &plusmn; 0.0147<span class="stat-label">against a pre-registered 3&sigma; threshold of 0.0439</span></span>

- Split the full test set by whether the caption carries spatial vocabulary, generated both halves from the same checkpoint, scored both.
- Spatial 0.5971, non-spatial 0.5901. The difference is smaller than one standard error, against a **3σ threshold fixed before the run**.
- The minimum effect this test could have resolved was 0.039 — about four times tighter than any earlier generation-side comparison here. This is a null with the power to mean something, not a null from a small sample.

::right::

<img src="/img/fig08_effect_vs_floor.svg" alt="The measured spatial-minus-non-spatial effect plotted against its own pre-registered 3-sigma threshold, showing the effect well inside the noise band">

<!--
deck/PRESENTER_NOTES.md / re-verified directly against docs/EXPERIMENT_DESIGN_E3.md lines
163-166 (the project's own S9.1 table): Guo +0.0070, SE 0.0146, +0.48 sigma against a
pre-registered 3 sigma threshold of 0.0439; MDE 0.039 spatial / 0.046 non-spatial. Exact match,
re-verified independently this session (not just taken from the plan). Denominator note
(2,688/1,920 effective vs 2,708/1,932 subset sizes) kept OFF the face per the plan's own
instruction -- moves SE by <0.0001, changes no conclusion, face carries no n at all rather than
a contested one.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Measurement: does a second, independent evaluator agree?

A result only one instrument can see is a property of that instrument until a second one is tried.

<div class="eyebrow">Per-sample agreement between the two evaluators, n = 4,640</div>
<span class="stat">r = 0.404<span class="stat-label">measured at full scale, not taken from a smaller sample</span></span>

- A second, independently trained evaluator scored the same motions and reversed the sign: −0.0263, z = −1.95, against the primary's +0.0070, z = +0.48.
- They agree in aggregate and share only 16% of their per-sample variance — overlapping instruments, not identical ones.
- Neither clears the pre-registered threshold; both are reported as a positive result would have been.

::right::

<img src="/img/fig09_two_evaluators.svg" alt="Both evaluators' effect sizes plotted on one shared axis with zero marked and confidence intervals drawn, straddling zero from opposite sides">

<!--
deck/PRESENTER_NOTES.md: -0.0263, SE 0.0134, -1.95 sigma, threshold 0.0403 --
docs/EXPERIMENT_DESIGN_E3.md:166, already independently verified earlier this project.

r=0.404 per-sample correlation, n=4,640 -- artifacts/e3/evaluator_correlation.json, computed by
scripts/e3_evaluator_correlation.py (reuses e3_score.py's validated embedding conventions, no
regeneration). r^2=0.163 -- the two evaluators share about 16% of per-sample variance, 84% not
shared, which is what "overlapping but not identical" means on the face.

Do NOT claim the spatial/non-spatial split differs in evaluator agreement. The record also has
spatial r=0.392, non-spatial r=0.421 -- independently re-verified via Fisher z on the actual
per-subset n (2,708 spatial / 1,932 non-spatial): z=(atanh(0.392)-atanh(0.421))/
sqrt(1/2705+1/1929) = -1.175, matching the planner's -1.17 exactly. Not significant, not on any
face, not in any bullet.

The withdrawn figure (r=0.328, notebooks/02, n=128 -- the same E0b cached generation whose own
R-Precision-top3 (0.7578) sits 3.71 sigma from the real n=4,640 value, docs/EXPERIMENT_DESIGN_E3.md
S9.5) flattered the deck's own argument rather than weakening it: 0.328 implies r^2=10.8%, making
the two evaluators look LESS alike than the real 16.3% -- an error that ran in the direction of
the conclusion, which is exactly the kind a motivated check does not catch. The rule this earns:
verify the sample a number came from, not just whether the conclusion still sounds right.

Trimmed from four bullets to three 2026-09-16 (Joel/planner) to give this figure -- the deck's
best -- vertical room; the dropped setup sentence is recoverable from the WHY-line and the
figure's own row labels. Combined with the wide-left column restructure, the image no longer
needs a tight per-slide cap at all.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Analysis: was that comparison fair to begin with?

The two halves were matched on nothing except the property under test.

<div class="eyebrow">How much longer spatial captions are</div>
<span class="stat">40%<span class="stat-label">14.31 vs 10.22 words on average</span></span>

- Spatial captions average 14.31 words against non-spatial 10.22. Longer captions are easier to retrieve against, so the spatial half started with an advantage.
- Controlled by splitting each half into length terciles. On the primary evaluator, length lifts the score for spatial captions (0.509 → 0.587 → 0.598) and does essentially nothing for non-spatial ones (0.572 / 0.544 / 0.569).
- On the second evaluator the same terciles are non-monotonic, so this pattern is a property of one instrument and is scoped to it here.
- The interaction was not predicted, and it is not explained here.

::right::

<img src="/img/fig10_terciles.svg" alt="Two lines showing R-Precision across caption-length terciles, spatial rising and non-spatial flat">

<!--
deck/PRESENTER_NOTES.md / re-verified directly against artifacts/e3/e3_record.json:
mean_caption_words 14.308 (rounds to 14.31) vs 10.224 (10.22); Guo terciles spatial
0.5089/0.5871/0.5982 (rounds to 0.509/0.587/0.598), non-spatial 0.5719/0.5437/0.5687
(0.572/0.544/0.569). Exact match to the digit, independently recomputed from the JSON, not
taken from the plan. "Scoped to the primary evaluator on purpose... on TMR the same terciles
are non-monotonic" -- per PRESENTER_NOTES.md; not independently re-derived this session (would
need the TMR tercile breakdown, not yet computed standalone) but the face's own wording already
scopes the claim correctly regardless, so this does not block assembly.
-->

---
layout: two-cols
layoutClass: wide-left
---

# Measurement: can FID arbitrate at these sample sizes?

The field's other standard metric was the obvious tiebreaker.

<div class="eyebrow">The floor, as samples fall from 2,320 to 941</div>
<span class="stat">0.0957 &rarr; 0.2573<span class="stat-label">real-vs-real FID floor, 30 trials each</span></span>

- FID estimates a 512×512 covariance — 131,328 parameters from n points. Below a few thousand samples that estimate is rank-deficient.
- Measured directly: real motion scored against real motion, which should score zero, floors at 0.0957 at n = 2,320 per side and 0.2573 at n = 941. Thirty trials each.
- Three FID values computed on one bit-identical set of 128 motions: 1.0731, 1.3997, 3.2909.
- The floor is larger than the effect being measured. FID cannot settle this question at this scale.

::right::

<img src="/img/fig11_fid_floor.svg" alt="Three measured real-vs-real FID floors with error bars at decreasing sample sizes, no fitted curve">

<!--
deck/PRESENTER_NOTES.md: real-vs-real floors 0.0957 (n=2,320/side), 0.1791 (n=1,378), 0.2573
(n=941), 30 trials each -- artifacts/e3/e3_record.json fid_results.*.real_vs_real_floor.
1.0731/1.3997/3.2909 on one bit-identical set of 128 motions -- docs/LANDMINES.md:487-489.
131,328 covariance parameters -- 512x513/2, arithmetic. No fitted curve, binding: this
project's own earlier C/n floor law claim ("fits to 0.5%" from two points) was contradicted by
50 trials finding a ~20% spread -- points and error bars only, per PRESENTER_NOTES.md's own
citation of that correction. All figures already independently verified earlier this project
(e3_record.json, docs/LANDMINES.md); not re-run fresh this session but consistent throughout.
-->

---
layout: two-cols
layoutClass: wide-left
---

# What this establishes, and what it does not

Saying what a result does not support is the most useful line on a research slide.

<div class="eyebrow">Effects excluded at 2&sigma;; anything smaller stays open</div>
<span class="stat">&ge; 0.117<span class="stat-label">the smallest effect this project can rule out here</span></span>

- Established: the text encoder under-separates spatial language, in every representation tested, on a frozen component that additional motion training cannot reach.
- Not established: that the deficit propagates into generated motion. Effects ≥ 0.175 are excluded at 3σ and ≥ 0.117 at 2σ. Smaller effects remain open.
- Once the hardware path was fixed, the decisive experiment became affordable — an overnight run — and it was declined. The effect size it targeted was the experiment's own noise reading rather than a hypothesis-motivated one.

::right::

<img src="/img/fig12_exclusion.svg" alt="Bar showing the region of effect sizes excluded at 2 and 3 sigma against the region that remains open">

<!--
deck/PRESENTER_NOTES.md / re-verified directly against docs/DECISIONS.md lines 103-104
(D-24/27/28): "effects >=0.175 are excluded at 3 sigma, >=0.117 at 2 sigma". Exact match. The
declined-experiment framing (overnight run affordable, declined because the target was the
experiment's own noise reading) also directly verified against the same decision record
earlier this project.
-->

---

# Where this goes next

An open question with a known price is a plan, not a gap.

<div class="eyebrow">One full replication; 76.6h for the published protocol</div>
<span class="stat">4.03h<span class="stat-label">measured wall-clock per replication, on MPS</span></span>

- Closing the FID half needs independent regenerations, not more resampling of the one that exists. Three replications is about 8.1 hours; the published twenty, about 76.6.
- A direction being looked at: encoder-capacity asymmetry. A September 2026 result trains a 5×6 grid of CLIP models and reports that oversized text encoders degrade zero-shot performance while specifically damaging cross-modal alignment.
- That bears directly on slide 7's own open question — whether a larger text encoder would separate spatial language better. The probe is cheap and those checkpoints are public, so it is the next measurement rather than a speculation.

<!--
deck/PRESENTER_NOTES.md / re-verified directly against docs/DECISIONS.md lines 61-65 (D-03):
"4.03h wall-clock on MPS" per replication, "~8.1h of additional MPS compute" for 3 total,
"~76.6h of additional MPS compute" for the full 20-replication protocol. Exact match (these are
numbers this session itself derived and committed earlier in this project, so this is a direct
self-check, not a fresh derivation). arXiv:2609.05730 citation not independently re-fetched
this session -- planner states it fetched the abstract and released data directly; flagged as
[not independently re-verified by implementer] rather than silently treated as checked.
-->
