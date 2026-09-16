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
DECK STATUS: content assembly in progress. Slides 1-5, 7-8, 10, 12-13 built from
deck/SLIDE_COPY.md + deck/PRESENTER_NOTES.md, verified against source independently before
being placed on a face (see per-slide notes below for exact citations). Slides 6 and 11 held
back on the planner's explicit instruction (known text-collision defect, fig11 provenance
issue being fixed). Slide 2's architecture diagram not yet built (implementer-authored,
scheduled last per the agreed build order). Slide 9 held pending planner's reworded copy for
the real full-scale correlation figure (see that slide's own note).
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

<div style="height:280px;display:flex;align-items:center;justify-content:center;border:1px dashed #ccc;color:#828891;font-family:'Geist Mono',monospace;font-size:0.7em;text-align:center;">
architecture diagram<br>(pending — implementer-authored,<br>last in build order)
</div>

<!--
deck/PRESENTER_NOTES.md: "No number on this face, deliberately. An earlier draft carried
'23.3K training samples,' taken from the author's own account of the earlier project rather
than from this repository. A grep across every markdown file here returns no such figure, so
it was removed rather than softened. The four resolution levels appear in the diagram as
structure, which the diagram may depict as a design fact without asserting a measurement. No
'failed', 'broken' or 'wrong' appears on this face or in this diagram -- standing editorial
rule, DECK_PLAN.md S2."

TODO(implementer): architecture diagram not yet built. Placeholder in the right column above.
-->

---

# Measurement: what the pipeline reports once it is instrumented

A model that produces plausible output and raises no exception cannot be judged by looking at it.

<div class="eyebrow">Pipeline and metric failure modes, verified not asserted</div>
<span class="stat">17<span class="stat-label">that change the answer and raise no error</span></span>

- Rebuilding with measurement at every stage produced a catalogue of silent failure modes in the pipeline and the metric stack, each verified rather than asserted.
- The load-bearing one is a decode check. Bone lengths in this dataset are constant by construction; measured under the original slice they vary by 25.81% on average, and up to 81.70% on individual bones.
- The number is not the finding. The invariant is — bone length is checkable with no model, no label and no metric.

<img src="/img/03_skeleton_side_by_side.png" alt="Skeleton decoded correctly next to the same motion decoded with the wrong slice, side by side">

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

# Measurement: does this harness reproduce a published number?

An instrument that has never agreed with an external reference is not yet an instrument.

<div class="eyebrow">The gap, against the published figure's own uncertainty</div>
<span class="stat">0.0062<span class="stat-label">vs the published figure's own &plusmn;0.007, n = 4,640</span></span>

- Generated the full test split from the released checkpoint and scored it: 0.6172 against the paper's own 0.611 ± 0.007. The difference is smaller than the reference's own stated uncertainty.
- This validates the whole path end to end — checkpoint loading, generation, scoring — not only the evaluator's ability to re-encode motion that was already correct.
- The FID half of the same check is a non-rejection, not a tight reproduction. That distinction is kept open rather than counted as met.

<div style="height:100px;display:flex;align-items:center;justify-content:center;border:1px dashed #ccc;color:#828891;font-family:'Geist Mono',monospace;font-size:0.7em;">
figure pending — regenerating with convention-free ±1 SE labelling
</div>

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

FIGURE HELD per planner: fig05_reproduction.svg is being regenerated with a convention-free
annotation and explicit +-1 SE labelling. Do not reference the old svg; placeholder above
until it lands, same treatment as slides 6 and 11.
-->

---

# Slide 6 — TBD (held back: text-collision fix in progress)

TBD

<!-- SLIDE 6 HELD BACK per planner's explicit instruction: known text-collision defect,
figure worker fixing. Do not build from the current SLIDE_COPY/PRESENTER_NOTES snapshot until
the planner re-locks it clean. -->

---

# Experiment: is the deficit made by pooling, or already inside the encoder?

The cheap explanation has to be eliminated before the expensive one is worth testing.

<div class="eyebrow">Representations tested; the gap survives in all of them</div>
<span class="stat">3 of 3<span class="stat-label">pooled, mean-over-tokens, most-divergent-token</span></span>

- The model reads one pooled vector per caption — that vector is the entire conditioning signal, so its geometry is exactly what the generator sees. If pooling were discarding the distinction, the per-token features would still carry it.
- They do not. Pooled, mean-over-tokens, and most-divergent-token each keep a significant gap after Bonferroni correction for the three comparisons.
- The orderings differ — the raw gap is largest at the token level, the standardised effect largest for pooled — so the claim is survival, not growth.
- This matters because it relocates the weakness onto a frozen component. Nothing in this pipeline updates the text encoder, so more motion data will not move it.

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

# Experiment: does the encoder deficit reach the generated motion?

A weakness in a component is not automatically a weakness in the system.

<div class="eyebrow">Measured difference, beside its own standard error</div>
<span class="stat" style="font-size:1.3em">+0.0070 &plusmn; 0.0147<span class="stat-label">against a pre-registered 3&sigma; threshold of 0.0439</span></span>

- Split the full test set by whether the caption carries spatial vocabulary, generated both halves from the same checkpoint, scored both.
- Spatial 0.5971, non-spatial 0.5901. The difference is smaller than one standard error, against a **3σ threshold fixed before the run**.
- The minimum effect this test could have resolved was 0.039 — about four times tighter than any earlier generation-side comparison here. This is a null with the power to mean something, not a null from a small sample.

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

# Slide 9 — TBD (held back: awaiting planner's reworded copy for the real r=0.4037 correlation)

TBD

<!--
HELD PENDING PLANNER SIGN-OFF. scripts/e3_evaluator_correlation.py's real n=4,640 result
(artifacts/e3/evaluator_correlation.json): pearson_r_overall=0.4037, spatial=0.3917,
non_spatial=0.4210 -- reported to the planner 2026-09-16, meaningfully higher than notebooks/02's
n=128 figure (0.328) that PRESENTER_NOTES.md previously (and incorrectly) cited for this slide.
Per the planner's own instruction ("send me the number before you put it on the face... if it
comes back much higher, slide 9's bullet needs rewording"), this slide is NOT assembled yet --
STAT and the correlation bullet wait for the planner's reworded text. The rest of the slide's
claim (E3 S9.1's -0.0263/z=-1.95 figure, the pre-registered-prominence commitment) is already
verified and ready; only the correlation figure and its bullet are pending.
-->

---

# Analysis: was that comparison fair to begin with?

The two halves were matched on nothing except the property under test.

<div class="eyebrow">How much longer spatial captions are</div>
<span class="stat">40%<span class="stat-label">14.31 vs 10.22 words on average</span></span>

- Spatial captions average 14.31 words against non-spatial 10.22. Longer captions are easier to retrieve against, so the spatial half started with an advantage.
- Controlled by splitting each half into length terciles. On the primary evaluator, length lifts the score for spatial captions (0.509 → 0.587 → 0.598) and does essentially nothing for non-spatial ones (0.572 / 0.544 / 0.569).
- On the second evaluator the same terciles are non-monotonic, so this pattern is a property of one instrument and is scoped to it here.
- The interaction was not predicted, and it is not explained here.

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

# Slide 11 — TBD (held back: figure provenance issue being fixed)

TBD

<!-- SLIDE 11 HELD BACK per planner's explicit instruction: fig11 currently carries a false
claim about its own provenance, figure worker fixing in place. Do not build until re-locked. -->

---

# What this establishes, and what it does not

Saying what a result does not support is the most useful line on a research slide.

<div class="eyebrow">Effects excluded at 2&sigma;; anything smaller stays open</div>
<span class="stat">&ge; 0.117<span class="stat-label">the smallest effect this project can rule out here</span></span>

- Established: the text encoder under-separates spatial language, in every representation tested, on a frozen component that additional motion training cannot reach.
- Not established: that the deficit propagates into generated motion. Effects ≥ 0.175 are excluded at 3σ and ≥ 0.117 at 2σ. Smaller effects remain open.
- Once the hardware path was fixed, the decisive experiment became affordable — an overnight run — and it was declined. The effect size it targeted was the experiment's own noise reading rather than a hypothesis-motivated one.

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
