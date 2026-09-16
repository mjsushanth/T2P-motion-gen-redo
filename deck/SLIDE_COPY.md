# SLIDE_COPY — exact face text, 13 slides

Planner-owned. C1 assembles this into `slides.md` with layout, stat components and figure
refs; do not rewrite the prose. If a slide overflows, tell me and I will cut words — do not
silently trim, and do not move a bullet into presenter notes without saying so.

Format per slide: TITLE / WHY-LINE (always shown, directly under the title, one sentence for
the non-specialist) / STAT / BULLETS / FIGURE.

---

## 1 — Cover

**T2P-Reboot**
*Taking a text-to-motion pipeline apart*

One released checkpoint · 4,640 generated sequences · one laptop

FIGURE: none. STAT: none. Leave it sparse.

---

## 2 — The predecessor system: conditioned diffusion over 3D human pose

WHY: A text-to-motion model turns a written sentence into a sequence of 3D body poses. This is the system the rest of the deck takes apart.

STAT: none.

> NOT FACE TEXT — verification note for the implementer. The predecessor's training-set size
> and head count are described by the author, not measured anywhere in this repository; a grep
> across every markdown file returns nothing. No unmeasured number goes on a face. The
> architecture diagram carries the four resolution levels as structure, which is a design fact
> it can depict without asserting a measurement.

- A UNet diffusion backbone conditioned on CLIP text embeddings, injected by cross-attention at four resolution levels — coarse layers carrying global posture, finer layers limb and hand placement.
- Anatomical plausibility enforced during training: bone-length consistency, joint limits, and kinematic-chain validation from pelvis through the extremities.
- Classifier-free guidance trading diversity against adherence to the prompt, ablated over guidance strength and noise schedule.

These are the right things to want from a text-to-pose model. The open question is how you would know whether you got them.

FIGURE: architecture diagram (right column).

---

## 3 — Measurement: what the pipeline reports once it is instrumented

WHY: Motion is stored as long vectors of numbers, not pictures. Read the wrong numbers out and you still get a skeleton — just not a human one, and nothing errors.

STAT: **17** — pipeline and metric failure modes that change the answer and raise no error

- Rebuilding with measurement at every stage produced a catalogue of silent failure modes in the pipeline and the metric stack, each verified rather than asserted.
- The load-bearing one is a decode check. Bone lengths in this dataset are constant by construction; measured under the original slice they vary by 25.81% on average, and up to 81.70% on individual bones.
- The number is not the finding. The invariant is — bone length is checkable with no model, no label and no metric.

FIGURE: `notebooks/03_skeleton_side_by_side.png`

---

## 4 — The question: what does a benchmark score certify?

A text-to-motion model is scored by a retrieval benchmark. The published figure for the reference model is **0.611**.

That number cannot, on its own, separate a model that followed the sentence from one that produced a plausible motion.

Everything that follows is an attempt to find out which.

FIGURE: none — argued exception. STAT: none; the 0.611 is inline and is the object of study, not a result.

---

## 5 — Measurement: does our own rebuilt pipeline land on the published number?

WHY: The reference model's paper reports a score of 0.611 on a standard retrieval benchmark. We rebuilt that scoring pipeline ourselves and ran it to see where we would land.

STAT: **0.0062** — the gap, against the published figure's own ±0.007, n = 4,640

- The benchmark asks one question: given a generated motion, does its own caption rank in the top 3 of a 32-candidate pool? Pure chance is 3 in 32, about 0.094.
- We took their released checkpoint, generated motion for all 4,640 test captions on a laptop, and scored it with a harness rebuilt here from scratch.
- It came out at 0.6172 against their 0.611 ± 0.007. Nothing was tuned toward their figure — we ran our own pipeline and arrived at it.
- Landing there independently is what makes the measurements later in this deck worth reading.

FIGURE: `fig05_reproduction.svg`

---

## 6 — Experiment: can the text encoder tell "left" from "right"?

WHY: CLIP is the frozen language model that turns a caption into the numbers the generator reads. If it cannot separate "left" from "right", nothing downstream can.

STAT: **d = 0.995** — spatial against comparable non-spatial contrasts

- Sentences differing only by *left* / *right* sit at 0.9707 cosine similarity. Sentences differing by a comparable non-spatial modifier, in the same syntactic slot, sit at 0.9442.
- Scale first: two sentences on entirely unrelated topics still score 0.72. This encoder compresses ordinary English into a narrow band near the top, so the question is never "is 0.97 high" — it is "is 0.97 higher than it should be."
- Sample size was fixed by a power calculation before the data, and the first and second batches of pairs are each significant on their own.
- 58.4% of the benchmark's captions contain spatial vocabulary.

FIGURE: `fig06_clip_separation.svg`

---

## 7 — Experiment: is the deficit made by pooling, or already inside the encoder?

WHY: CLIP produces one vector per word, then averages them into a single vector for the generator. Perhaps the averaging is what loses direction — so we looked at the per-word vectors directly.

STAT: **3 of 3** — representations tested; the gap survives in all of them

- That pooled vector is the entire conditioning signal — its geometry is exactly what the generator sees.
- They do not. Pooled, mean-over-tokens, and most-divergent-token each keep a significant gap after Bonferroni correction for the three comparisons.
- The gap does not grow in a straight line across the three, so the claim is survival, not growth.
- This matters because it relocates the weakness onto a frozen component. Nothing in this pipeline updates the text encoder, so more motion data will not move it.

FIGURE: `fig07_pooling.svg`

---

## 8 — Experiment: does the encoder deficit reach the generated motion?

WHY: The encoder is weak on spatial words. That does not automatically make the motions worse, so we generated both halves of the test set and compared them.

STAT: **+0.0070 ± 0.0147** — the measured difference, beside its own standard error

- Split the full test set by whether the caption carries spatial vocabulary, generated both halves from the same checkpoint, scored both.
- Spatial 0.5971, non-spatial 0.5901. The difference is smaller than one standard error, against a **3σ threshold fixed before the run**.
- The minimum effect this test could have resolved was 0.039 — about four times tighter than any earlier generation-side comparison here. This is a null with the power to mean something, not a null from a small sample.

FIGURE: `fig08_effect_vs_floor.svg`

---

## 9 — Measurement: does a second, independent evaluator agree?

WHY: That score depends on which retrieval model does the grading. A second, independently trained grader exists, so we ran the same motions through it as well.

STAT: **r = 0.404** — per-sample agreement between the two evaluators, n = 4,640

- A second, independently trained evaluator scored the same motions and reversed the sign: −0.0263, z = −1.95, against the primary's +0.0070, z = +0.48.
- They agree in aggregate and share only 16% of their per-sample variance — overlapping instruments, not identical ones.
- Neither clears the pre-registered threshold; both are reported as a positive result would have been.

> NOT FACE TEXT — cut from four bullets to three on 2026-09-16 to give fig09 vertical room.
> It is the deck's strongest figure and was rendering at roughly 16% of canvas width. The
> dropped setup sentence is recoverable from the WHY-line and the figure's own row labels.

FIGURE: `fig09_two_evaluators.svg`

---

## 10 — Analysis: was that comparison fair to begin with?

WHY: Captions containing spatial words are also longer captions, and longer captions are easier to retrieve. The previous comparison may have been unfair from the start.

STAT: **40%** — how much longer spatial captions are

- Spatial captions average 14.31 words against non-spatial 10.22, so the spatial half started with an advantage.
- Controlled by splitting each half into length terciles. On the primary evaluator, length lifts the score for spatial captions (0.509 → 0.587 → 0.598) and does essentially nothing for non-spatial ones (0.572 / 0.544 / 0.569).
- On the second evaluator the same terciles are non-monotonic, so this pattern is a property of one instrument and is scoped to it here.
- The interaction was not predicted, and it is not explained here.

FIGURE: `fig10_terciles.svg`

---

## 11 — Measurement: can FID arbitrate at these sample sizes?

WHY: FID is the field's other standard score. It compares the statistical shape of a set of generated motions against real ones, and it was the obvious way to break the tie.

STAT: **0.0957 → 0.2573** — the floor, as samples fall from 2,320 to 941

- FID estimates a 512×512 covariance — 131,328 parameters from n points. Below a few thousand samples that estimate is rank-deficient.
- Measured directly: real motion scored against real motion, which should score zero, floors at 0.0957 at n = 2,320 per side and 0.2573 at n = 941. Thirty trials each.
- Three FID values computed on one bit-identical set of 128 motions: 1.0731, 1.3997, 3.2909.
- The floor is larger than the effect being measured. FID cannot settle this question at this scale.

FIGURE: `fig11_fid_floor.svg`

---

## 12 — What this establishes, and what it does not

WHY: What the measurements support, what they rule out, and what is still genuinely open.

STAT: **≥ 0.117** — effects excluded at 2σ; anything smaller stays open

- Established: the text encoder under-separates spatial language, in every representation tested, on a frozen component that additional motion training cannot reach.
- Not established: that the deficit propagates into generated motion. Effects ≥ 0.175 are excluded at 3σ and ≥ 0.117 at 2σ. Smaller effects remain open.
- Once the hardware path was fixed, the decisive experiment became affordable — an overnight run — and it was declined. The effect size it targeted was the experiment's own noise reading rather than a hypothesis-motivated one.

FIGURE: `fig12_exclusion.svg`

---

## 13 — Where this goes next

WHY: The one piece left unresolved, what it would cost to close, and a result published last week that bears on it.

STAT: **4.03h** — one full replication; 76.6h for the published protocol

- Closing the FID half needs independent regenerations, not more resampling of the one that exists. Three replications is about 8.1 hours; the published twenty, about 76.6.
- A direction being looked at: a September 2026 result trains a 5×6 grid of CLIP models and reports that oversized text encoders degrade performance, damaging cross-modal alignment specifically.
- That bears directly on slide 7's own open question — whether a larger text encoder would separate spatial language better. The probe is cheap and those checkpoints are public, so it is the next measurement rather than a speculation.

FIGURE: none, or a three-rung cost ladder. Argue the exception.
