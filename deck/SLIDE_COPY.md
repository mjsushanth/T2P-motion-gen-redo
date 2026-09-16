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

WHY: Before measuring a pipeline, you have to see what it was built to do.

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

WHY: A model that produces plausible output and raises no exception cannot be judged by looking at it.

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

## 5 — Measurement: does this harness reproduce a published number?

WHY: An instrument that has never agreed with an external reference is not yet an instrument.

STAT: **0.0062** — the gap, against the published figure's own ±0.007, n = 4,640

- Generated the full test split from the released checkpoint and scored it: 0.6172 against the paper's own 0.611 ± 0.007. The difference is smaller than the reference's own stated uncertainty.
- This validates the whole path end to end — checkpoint loading, generation, scoring — not only the evaluator's ability to re-encode motion that was already correct.
- The FID half of the same check is a non-rejection, not a tight reproduction. That distinction is kept open rather than counted as met.

FIGURE: `fig05_reproduction.svg`

---

## 6 — Experiment: can the text encoder tell "left" from "right"?

WHY: If the conditioning signal never carries direction, nothing downstream can recover it.

STAT: **d = 0.995** — spatial against comparable non-spatial contrasts

- Sentences differing only by *left* / *right* sit at 0.9707 cosine similarity. Sentences differing by a comparable non-spatial modifier, in the same syntactic slot, sit at 0.9442.
- Scale first: two sentences on entirely unrelated topics still score 0.72. This encoder compresses ordinary English into a narrow band near the top, so the question is never "is 0.97 high" — it is "is 0.97 higher than it should be."
- Sample size was fixed by a power calculation run before the full data. The pilot and the extension are each independently significant, which forecloses the objection that easier pairs were written the second time.
- 58.4% of the benchmark's captions contain spatial vocabulary.

FIGURE: `fig06_clip_separation.svg`

---

## 7 — Experiment: is the deficit made by pooling, or already inside the encoder?

WHY: The cheap explanation has to be eliminated before the expensive one is worth testing.

STAT: **3 of 3** — representations tested; the gap survives in all of them

- The model reads one pooled vector per caption — that vector is the entire conditioning signal, so its geometry is exactly what the generator sees. If pooling were discarding the distinction, the per-token features would still carry it.
- They do not. Pooled, mean-over-tokens, and most-divergent-token each keep a significant gap after Bonferroni correction for the three comparisons.
- The orderings differ — the raw gap is largest at the token level, the standardised effect largest for pooled — so the claim is survival, not growth.
- This matters because it relocates the weakness onto a frozen component. Nothing in this pipeline updates the text encoder, so more motion data will not move it.

FIGURE: `fig07_pooling.svg`

---

## 8 — Experiment: does the encoder deficit reach the generated motion?

WHY: A weakness in a component is not automatically a weakness in the system.

STAT: **+0.0070 ± 0.0147** — the measured difference, beside its own standard error

- Split the full test set by whether the caption carries spatial vocabulary, generated both halves from the same checkpoint, scored both.
- Spatial 0.5971, non-spatial 0.5901. The difference is smaller than one standard error, against a **3σ threshold fixed before the run**.
- The minimum effect this test could have resolved was 0.039 — about four times tighter than any earlier generation-side comparison here. This is a null with the power to mean something, not a null from a small sample.

FIGURE: `fig08_effect_vs_floor.svg`

---

## 9 — Measurement: does a second, independent evaluator agree?

WHY: A result only one instrument can see is a property of that instrument until a second one is tried.

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

WHY: The two halves were matched on nothing except the property under test.

STAT: **40%** — how much longer spatial captions are

- Spatial captions average 14.31 words against non-spatial 10.22. Longer captions are easier to retrieve against, so the spatial half started with an advantage.
- Controlled by splitting each half into length terciles. On the primary evaluator, length lifts the score for spatial captions (0.509 → 0.587 → 0.598) and does essentially nothing for non-spatial ones (0.572 / 0.544 / 0.569).
- On the second evaluator the same terciles are non-monotonic, so this pattern is a property of one instrument and is scoped to it here.
- The interaction was not predicted, and it is not explained here.

FIGURE: `fig10_terciles.svg`

---

## 11 — Measurement: can FID arbitrate at these sample sizes?

WHY: The field's other standard metric was the obvious tiebreaker.

STAT: **0.0957 → 0.2573** — the floor, as samples fall from 2,320 to 941

- FID estimates a 512×512 covariance — 131,328 parameters from n points. Below a few thousand samples that estimate is rank-deficient.
- Measured directly: real motion scored against real motion, which should score zero, floors at 0.0957 at n = 2,320 per side and 0.2573 at n = 941. Thirty trials each.
- Three FID values computed on one bit-identical set of 128 motions: 1.0731, 1.3997, 3.2909.
- The floor is larger than the effect being measured. FID cannot settle this question at this scale.

FIGURE: `fig11_fid_floor.svg`

---

## 12 — What this establishes, and what it does not

WHY: Saying what a result does not support is the most useful line on a research slide.

STAT: **≥ 0.117** — effects excluded at 2σ; anything smaller stays open

- Established: the text encoder under-separates spatial language, in every representation tested, on a frozen component that additional motion training cannot reach.
- Not established: that the deficit propagates into generated motion. Effects ≥ 0.175 are excluded at 3σ and ≥ 0.117 at 2σ. Smaller effects remain open.
- Once the hardware path was fixed, the decisive experiment became affordable — an overnight run — and it was declined. The effect size it targeted was the experiment's own noise reading rather than a hypothesis-motivated one.

FIGURE: `fig12_exclusion.svg`

---

## 13 — Where this goes next

WHY: An open question with a known price is a plan, not a gap.

STAT: **4.03h** — one full replication; 76.6h for the published protocol

- Closing the FID half needs independent regenerations, not more resampling of the one that exists. Three replications is about 8.1 hours; the published twenty, about 76.6.
- A direction being looked at: encoder-capacity asymmetry. A September 2026 result trains a 5×6 grid of CLIP models and reports that oversized text encoders degrade zero-shot performance while specifically damaging cross-modal alignment.
- That bears directly on slide 7's own open question — whether a larger text encoder would separate spatial language better. The probe is cheap and those checkpoints are public, so it is the next measurement rather than a speculation.

FIGURE: none, or a three-rung cost ladder. Argue the exception.
