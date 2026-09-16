# DECK_PLAN — T2P-Reboot presentation deck

**Canonical record.** Messages are notifications; this file is the ruling. Where a message
and this document disagree, this document wins.

**Ownership.** Planner (Opus review session) owns this file, the narrative, the numbers and
the figure specs. Implementer (C1) owns `deck/slides.md`, `deck/style.css`, the build, the
deploy and all rendering verification. Neither edits the other's files without announcing a
re-lock.

**Standing instruction to the implementer: verify my figures rather than take them.** Every
number below carries a source. Open the source. A spec that propagates a planner's memory
error is worse than no spec.

---

## 1. What this deck is about

**The frame is a question, not a proposition.** Text-to-motion models are scored on a
benchmark. That benchmark produces a number. The question this deck asks is what that number
actually certifies — and the work is a sequence of experiments, each one motivated by the
result of the previous.

**Audience: balanced, HR and researcher.** Practical consequence per slide — the claim line
and the intuition are legible to a non-specialist; one or two bullets carry the depth a
researcher is reading for; the figure does the work in between. Never both registers in the
same sentence.

**The one finding to be remembered:** *the text encoder cannot reliably tell "left" from
"right", and because it is frozen, no amount of additional training on motion data can fix it.*
That is the memorable claim. The deck's depth signal is different and arrives later — that the
project then **tested whether that deficit propagates to generated motion and declined to claim
that it does.**

---

## 2. Framing the predecessor — binding editorial rule

The 2025 project is **not** introduced as a failure, and no slide, title, bullet or figure
caption uses "failed", "broken", "wrong" or any synonym about it.

It was a conditioned cross-modal diffusion architecture built in an early era, reaching for
things that were genuinely hard and are genuinely correct to want: anatomical plausibility
enforced during training, kinematic-chain consistency, classifier-free guidance for
conditioning strength, hierarchical semantic conditioning across resolutions. **Those concepts
are valid and hard-won.** Research advances by models that were reasonable, got cited, and were
superseded by deeper ones. That is the normal shape of the field, not a defect.

What the rebuild supplies is **instrumentation** — the measurement layer that was not there.
Frame every predecessor claim as *"instrumenting this produced X"*, never as *"this was wrong."*

---

## 3. Title convention

Plain technical register, typed the way the repository README types its entries. No slogans,
no rhetorical questions used as sermon, no "the surprising truth about" constructions.

```
Experiment  — <what was done>
Measurement — <what was quantified>
Analysis    — <what was decomposed>
```

Each slide face carries **one line stating why that experiment is worth running** directly
under the title. That line is for the non-specialist and it is not optional.

---

## 4. Slide budget — 13

Cover, two predecessor slides, ten Spine A slides. No section dividers; at this budget they
cost more than they organise.

**Density gradient.** Slide 1 near-empty. Slides 2-5 one measurement each. Slides 9-10 are
where the depth lands. Depth is still arriving at slide 12. No slide anywhere claims rigour —
it states what was done and lets the reader conclude it.

---

## 5. The slides

Legend for verification status:
`[V]` verified this session against the cited source · `[D]` derived by the planner, must be
independently re-derived before it goes on a face · `[U]` not yet opened, implementer must read
before building

### 1 — Cover

**T2P-Reboot** · *Taking a text-to-motion pipeline apart*

Scope line, bare and factual: one released checkpoint, one benchmark, 4,640 generated
sequences, one laptop. No figure. No stat component.

### 2 — The predecessor system: conditioned diffusion over 3D human pose

**Why this slide exists:** the reader needs the object before any measurement of it means anything.

- Claim: a cross-modal diffusion architecture that conditioned a UNet on CLIP text embeddings
  through multi-resolution cross-attention, with anatomical plausibility enforced in the
  training objective and classifier-free guidance controlling conditioning strength.
- Number: **none — deliberately.** `[REJECTED]` The first draft used "23.3K training samples,"
  taken from the author's own account rather than from this repository. A grep across every
  markdown file in the repo returns no such figure. It is removed rather than softened: an
  unmeasured number on a face is the exact failure `verification-doctrine.md` §1 describes, and
  the rule there is to rewrite the claim so it does not need the figure. Slide 2 now carries no
  stat; the diagram shows the four resolution levels as structure, not as a measurement.
- Figure: **architecture diagram** — the only structural diagram in the deck. Text encoder →
  projection → cross-attention injection points at four resolutions → diffusion backbone →
  pose output, with the auxiliary loss terms annotated where they attach.
- Tone: this is what the system was reaching for. State the ambitions as ambitions.

### 3 — Measurement: what the pipeline reports once it is instrumented end to end

**Why this slide exists:** a model that produces plausible output and no error message cannot be
evaluated by looking at it.

- Claim: rebuilding the pipeline with measurement at every stage produced a catalogue of
  failure modes that raise no exception and change the answer.
- Number: **17 documented failure modes** `[V — docs/LANDMINES.md §1-15, §24, §25]`. **Corrected
  down from 25.** The document has 25 numbered sections, but §16-§23 are review-discipline
  lessons about producing research, not traps in the motion pipeline or the metric stack — the
  document says so itself under its own "Review-discipline lessons" heading. Slide 3's claim is
  about the pipeline, so 17 is the defensible count. Of these, the
  load-bearing one is a pose-decode check: bone lengths under the original slice vary by
  **25.81% mean CV, up to 81.70% on individual bones**, where a correctly decoded skeleton is
  constant by construction `[V — FORENSICS.md F1 Result]`
- Figure: **skeleton side-by-side** (`notebooks/03_skeleton_side_by_side.png` exists) — instantly
  legible to a non-specialist, precise for a researcher.
- Editorial: "instrumenting this produced" — not "this was broken."

### 4 — The question: what does a benchmark score certify?

**Why this slide exists:** every number that follows is an attempt to answer this.

- Claim: a text-to-motion benchmark score cannot, on its own, distinguish a model that followed
  the sentence from one that produced a plausible motion.
- Number: **0.611** — the published reference figure, presented as the object of study, not as a
  result `[V — e3_record.json, published_reproduction.published_value]`
- Figure: **none.** The argued exception. A figure here would establish credibility before the
  question has been asked, which reads as anxiety.

### 5 — Measurement: does this harness reproduce a published number?

**Why this slide exists:** an instrument that has never agreed with an external reference is not
an instrument.

- Claim: the harness reproduces the published figure on a checkpoint it did not train, at full
  scale.
- Number: **0.6172 vs 0.611 ± 0.007 — 0.87σ, n = 4,640**
  `[V — e3_record.json, published_reproduction]`
- Figure: **interval-overlap plot.** Two point estimates, two real standard errors (0.007
  published, 0.0072 binomial). Nothing drawn that was not measured.

### 6 — Experiment: can the text encoder distinguish "left" from "right"?

**Why this slide exists:** if the conditioning signal never carries direction, no downstream
model can recover it.

- Claim: sentences differing only by a spatial word land measurably closer together than
  sentences differing by a comparable non-spatial word in the same syntactic slot.
- Number: **0.9707 vs 0.9442, Cohen's d = +0.995** `[V — notebook 01 executed output]`
- Depth bullets (researcher register): sample size fixed by a power calculation run before the
  full data (n=40/group, achieved power 0.913); clears a Bonferroni threshold of 0.0167 across
  three comparisons; pilot (n=16) and extension (n=24) halves each independently significant,
  which forecloses "easier pairs were written the second time." `[V — notebook 01]`
- Intuition line (HR register): 1.00 means identical. Two sentences about completely different
  topics score 0.72. CLIP compresses ordinary English into a narrow band near the top, so the
  question is never "is 0.97 high" — it is "is 0.97 higher than it should be."
- Figure: **paired strip plot**, 40 points per group. These are real observations, so points are
  honest here.
- Scope, stated as scope not confession: **58.4%** of the benchmark's captions contain spatial
  vocabulary `[V — e3_record.json, 2,708/4,640]`

### 7 — Experiment: is that deficit created by pooling, or already inside the encoder?

**Why this slide exists:** the cheap explanation has to be eliminated before the expensive one is
worth testing.

- Claim: the gap survives in every representation tested — it does not vanish when pooling is
  removed.
- Number: three representations, all clearing Bonferroni — pooled **gap +0.0265, d = +0.995**;
  mean-over-tokens **+0.0185, d = +0.589**; most-divergent token **+0.1106, d = +0.724**
  `[V — notebook 01b executed output]`
- **Planner correction, do not restore the earlier wording.** An earlier draft of this plan said
  "the gap widens." It does not widen monotonically: the raw gap is largest at the token level
  but the *standardised* effect is largest for pooled, and the mean-token representation is the
  weakest of the three. The defensible claim is survival, not growth. Draw the raw gap and the
  effect size as two encodings on the same figure so the reader sees both orderings.
- Figure: **3-bar comparison** (`notebooks/01b_pooling_gap_by_representation.png` exists).
  Degenerate by design; three bars for three representations is the honest encoding.
- This is the slide that makes the sequence a sequence. Say plainly that this experiment exists
  because of the previous result.

### 8 — Experiment: does the encoder deficit reach the generated motion?

**Why this slide exists:** a weakness in a component is not automatically a weakness in the system.

- Claim: on the primary evaluator, motion generated from spatial captions scores no worse than
  motion generated from non-spatial ones.
- Number: **+0.0070, z = 0.48** (n = 2,688 vs 1,920) `[V — planner derived from e3_record.json,
  then re-derived independently under both pooled and unpooled two-proportion SE: z = +0.476 and
  +0.477]`
- **Denominator correction, binding on slides 8, 9 and 10.** The subset sizes are 2,708 and
  1,932, but R-Precision only scores *complete* 32-candidate batches, so the true scoring
  denominators are **2,688 and 1,920** — 20 spatial and 12 non-spatial samples are never scored.
  Confirmed by recovering exact integer success counts (1,605/2,688 and 1,133/1,920). Quoting
  2,708/1,932 beside a rate computed over 2,688/1,920 pairs numbers from two different
  populations. Use the effective n wherever an n appears next to a rate.
- Figure: **effect drawn against its own noise floor.** The floor is the point. Without it a
  reader cannot tell a null from a small win.
- Do not put a large accent number on this slide. A giant `0.48σ` looks like a result and means
  the opposite. Stat slot carries the effect and the floor as a pair.

### 9 — Measurement: does a second, independent evaluator agree?

**Why this slide exists:** a result that only one instrument can see is a property of the
instrument until a second one is tried.

- Claim: an independent evaluator puts the effect in the **opposite** direction, at marginal
  significance.
- Number: **−0.0263, z = −1.95** (same effective n) `[V — re-derived: −1.948 unpooled,
  −1.939 pooled]`
- **The number that makes this slide land**, and it was missing from the first draft: the two
  evaluators' per-sample scores correlate at only **r = 0.328** `[V — notebook 02 executed
  output]`. They agree on the aggregate and disagree sample by sample. That is what licenses
  treating the sign flip as instrument-dependence rather than noise.
- Figure: **both effects on one axis, zero marked, both confidence intervals drawn.** A single
  shared axis is the only honest encoding — two separate charts would conceal that the estimates
  straddle zero from opposite sides.
- This is the strongest slide in the deck. Two instruments, one dataset, opposite signs, neither
  conclusive. Let it sit without editorialising.

### 10 — Analysis: was that comparison fair to begin with?

**Why this slide exists:** the groups were not matched on anything except the property being tested.

- Claim: spatial captions are 40% longer, and caption length predicts score for spatial captions
  while doing essentially nothing for non-spatial ones.
- Number: **14.31 vs 10.22 words**; spatial terciles **0.509 → 0.587 → 0.598**, non-spatial
  **0.572 / 0.544 / 0.569** `[V — e3_record.json, terciles]`
- Figure: **two tercile lines on shared axes.** Three measured points per line, drawn as points.
  No smoothing, no fit.
- The interaction was not predicted. Say so.

### 11 — Measurement: can FID arbitrate at these sample sizes?

**Why this slide exists:** the field's other standard metric was the obvious tiebreaker, and it
cannot do the job here.

- Claim: FID's own floor at affordable sample sizes is larger than the effect being measured.
- Number: real-vs-real floor **0.0957** at n=2,320/side → **0.2573** at n=941, 30 trials each
  `[V — e3_record.json, fid_results.*.real_vs_real_floor]`
- Supporting: three FID values computed on one bit-identical set of 128 motions — **1.0731 /
  1.3997 / 3.2909** `[V — docs/LANDMINES.md §14]`
- Figure: **three measured floors with error bars. No fitted curve.** Binding constraint: an
  earlier `C/n` law claimed by this project to fit "to 0.5%" from two points was contradicted by
  50 trials spanning roughly 20%. Points and spread only.

### 12 — What this establishes, and what it does not

**Why this slide exists:** stating what a result does not support is the most credible thing a
research deck can do.

- Claim: the encoder deficit is measured and real; its propagation into generated motion is
  **bounded, not demonstrated.**
- Number: effects **≥ 0.175 excluded at 3σ, ≥ 0.117 at 2σ** at the sample size run
  `[V — docs/DECISIONS.md D-24/27/28]`
- Figure: **exclusion-bound bar** — the region ruled out against the region still open, drawn at
  its real width.
- Include the decision that is better than most positive results: once the hardware path was
  fixed the decisive experiment became affordable — an overnight run — and it was **declined**,
  because the effect size it targeted was the experiment's own noise reading rather than a
  hypothesis-motivated one `[V — docs/DECISIONS.md D-24/27/28]`.

### 13 — Where this goes next

**Why this slide exists:** an open question with a known price is a plan, not a gap.

- Claim: the unresolved half has a measured cost, and the encoder question has a live direction
  in the current literature.
- Number: **4.03h per full replication**; ~8.1h for a three-replication spread; ~76.6h for the
  published 20-replication protocol `[V — docs/DECISIONS.md D-03]`
- Forward-looking, in Joel's own framing — directions being looked at, not commitments:
  encoder-capacity asymmetry (arXiv:2609.05730, released 4 Sep 2026, trains a 5x6 grid of CLIP
  models and finds oversized text encoders *degrade* zero-shot performance while damaging
  cross-modal alignment specifically) bears directly on this deck's slide 7, whose own stated
  open question was whether a larger CLIP would separate spatial language better. `[V — arXiv
  abstract and repo data fetched this session]`
- Figure: none, or a three-rung cost ladder. Argue the exception — three numbers in a sentence
  beat a chart of three numbers.

---

## 6. Figure inventory

| Slide | Figure | Status |
| --: | :-- | :-- |
| 2 | architecture diagram | new, to build |
| 3 | skeleton side-by-side | exists, `notebooks/03_skeleton_side_by_side.png` |
| 5 | interval-overlap plot | new, to build |
| 6 | paired strip plot | new; `notebooks/01_clip_spatial_blindness_histogram.png` may substitute |
| 7 | 3-bar representation comparison | exists, `notebooks/01b_pooling_gap_by_representation.png` |
| 8 | effect vs noise floor | new, to build |
| 9 | two effects, shared axis, zero marked | new, to build — **highest value figure in the deck** |
| 10 | two tercile lines, shared axes | new, to build |
| 11 | three FID floors with error bars | new, to build |
| 12 | exclusion-bound bar | new, to build |

Ten figures across thirteen slides. Deliberately one structural diagram and nine chart-like —
a research deck's architecture diagram is cheap, and the arguments here are comparisons.

---

## 7. Numbers that must NOT be used without reconciliation

`RESULTS.md` predates the MPS work and the E3 run. It is stale in three specific ways and it is
the file a reader reaches first:

1. It prices generation at "~39 minutes per 128 samples on this CPU", "5 CPU-hours", "roughly
   100 CPU-hours" for 20 replications. **Superseded** by `DECISIONS.md` D-03: 4.03h per full
   4,640-sample replication on MPS, ~76.6h for 19 more.
2. It states the published generated-model number "did not succeed at any sample size this
   hardware could afford." **Superseded** — E3 reproduced the R-Precision half at 0.87σ.
3. Ground-truth R-Precision appears as both **0.7969** (E0b) and **0.7202** (E0a). Resolved as a
   hand-rolled loader bug in E0a's script; both values remain live in the repo. Cite E0b only.

One open conflict the planner could not resolve: `DECISIONS.md` D-24 states the MPS training
speedup as **9.95x**; an independent measurement in the planner's own session gave **9.89x**.
**Neither goes on a slide until one is re-measured.** Neither is load-bearing for any slide above.

---

## 8. Verification split

- **Implementer verifies:** rendering, per-slide overflow programmatically at every reachable
  animation state, deploy, and every `[V]` figure against its cited source.
- **Planner verifies:** narrative, claim-to-number fit, figure honesty, and re-derivation of the
  two `[D]` values.
- **Closed since the first draft:** notebook 01b's pooling figures and notebook 02's
  cross-evaluator correlation are both read and cited above.
- **Still uncovered by anyone:** the 9.95x / 9.89x MPS conflict in §7, and whether notebook 01's
  spatial-vocabulary detector and E3's are the same detector (both report ~58%, independently).

---

## 9. Hostile review — what a reviewer who knows this field attacks first

Run as a deliberate pass against the assembled spine. Ordered by how much damage each one does.

**A1 — "CLIP was never trained to make text-to-text similarity meaningful."**
The strongest attack available, and the first draft of the deck was silent on it. CLIP's
contrastive objective aligns text with *images*, not text with text. Measuring cosine between two
text embeddings is, on its face, an off-manifold use of the model, and a reviewer who knows the
architecture goes straight here.

**The answer is good, and it is now on slide 7.** This pipeline conditions the generator on the
pooled CLIP text vector — notebook 01b labels that representation "pooled (what MDM uses)". That
pooled vector is the entire conditioning signal the generator ever receives. So text-to-text
geometry in that space is not an off-label probe of CLIP; it is a direct measurement of the only
thing the generator is given. Slide 7 now states this in the first bullet. Do not cut it for
space.

**A2 — "Forty hand-written minimal pairs is an anecdote set, not a benchmark."**
Partly answered on the face already: the sample size was fixed by a power calculation before the
data, and the pilot and extension halves replicate independently. What remains true and
unanswered is that the pairs were authored by the same person testing them. The corpus incidence
figure (58.4%) shows the phenomenon is common in the benchmark; it does not show these particular
sentences are representative of it. **Presenter-notes answer, not a face answer** — putting it on
the face would spend slide 6's last words defending rather than stating.

**A3 — "The null on slide 8 could just be that the model has no headroom to lose."**
Answerable with a number already in hand: overall top-3 is 0.617 against a ground-truth ceiling
near 0.797 on the same instrument. There is roughly 0.18 of headroom, so a spatial deficit had
room to show up and did not. **Keep this in presenter notes for slide 8** and be ready to say it
aloud; it is the most likely live question.

**A4 — "Keyword matching conflates captions containing 'left' with captions requiring spatial
reasoning."** True, and not fully answerable. The split is lexical. A caption can contain a
directional word incidentally, and one can require spatial reasoning without using one. The
tercile control handles length, not this. **Name it in presenter notes as a live limitation** —
it is the cleanest thing a reviewer could ask to see improved, and conceding it costs nothing.

**A5 — "You are reporting a marginal −1.95σ you would have dismissed in the other direction."**
Fully answered by the pre-registered 3σ threshold, which slide 8 now states before slide 9 shows
the result. This is exactly why the pre-registration belongs on the earlier slide: it has to be
visible *before* the reader sees a number they might suspect was chosen after the fact.

**A6 — "Two evaluators disagreeing means one is broken, not that the result is instrument-
dependent."** Partly right, and the deck should not overclaim. The r = 0.328 per-sample
correlation on slide 9 is what licenses the weaker, defensible reading: they measure overlapping
but non-identical things. The face does not claim either evaluator is correct, and it should not
start.
