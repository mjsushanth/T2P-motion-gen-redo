# RESULTS

**What this project found, written for someone who has read nothing else in this repository.**
If a claim below needs another file to make sense, that is a bug in this document — say so.

Everything here comes from a solo, self-funded rebuild of a failed graduate course project on
text-to-motion generation, done on a single Apple Silicon laptop with no GPU and no budget. That
constraint is not an excuse; it is the actual question this project answers (see "What it cost,"
at the end): what can one person, alone, with a laptop, actually establish about a failed
research project and a real research question in a few days of work?

---

## 1. What was established

### 1.1 The original project's core numerical pipeline was broken from its first step

**Eight findings, verified across three phases of scrutiny: F1-F4 empirically, in this project's
own Stage 1 forensics run; F5 by direct code inspection; F6-F8 by a later, independent audit
that re-derived the algebra and re-read the source rather than taking the original report's own
framing at face value.**

- **F1 — the decode bug, the root cause underneath most of the rest.** The predecessor decoded
  3D pose vectors incorrectly — pulling the wrong 66 numbers out of every 263-number motion
  frame. Confirmed both by reading the official dataset's own decode logic and by directly
  measuring that skeletons built from the wrong slice don't have consistent bone lengths (real
  skeletons do, by construction, once decoded correctly). This single defect explains essentially
  every downstream anomaly the original project's own notes described — invented "coordinate
  system" quirks, joints sinking through the floor, odd hip offsets — none of which needed a new
  explanation once the decode was fixed.
- **F2 — a wiring bug independent of F1.** The unsupervised clustering used to balance training
  data was fit on different, mismatched data from what the model actually trained on.
- **F3 — the task design itself capped what was learnable.** Predicting a single static pose from
  only a caption's first clause, using literally the first frame of a clip as the "answer,"
  mostly taught the model to output a generic standing pose regardless of what the sentence
  described, because frame 0 is usually close to a generic standing pose no matter the caption.
- **F4 — no way to know any of the above, because nothing was measured.** No validation loop, no
  held-out test split, no fixed random seed, no experiment tracking anywhere in either notebook.
- **F5 — the engineering state made the other four hard to catch.** The entire system lived in
  four monolithic notebook cells; the same core classes were re-implemented three times, near-
  duplicated, in one file; no package, no config objects, no tests, no CLI, no seeds, hardcoded
  Windows paths. A codebase shaped like this resists the kind of inspection that would have
  caught F1-F4 sooner.
- **F6 — classifier-free guidance (a sampling-time technique) was implemented inside the training
  loss itself**, not at inference time. Worked through algebraically: if the model's conditional
  and unconditional predictions become identical, the training loss is *exactly zero regardless
  of the guidance strength* — the objective is fully satisfiable by a model that ignores its text
  conditioning entirely.
- **F7 and F8 — two further bugs that corrupted the auxiliary loss terms F6's objective still
  relied on:** batch statistics were used to normalize the training target instead of fixed
  dataset statistics (recomputed per batch, so identical poses in different batches received
  different training targets), and a single timestep was applied to an entire batch when
  timesteps were meant to vary per sample.

Because there was never a validation loop, a held-out test split, a fixed random seed, or any
experiment tracking (F4), the original project's headline number — a "99.995% loss reduction"
across three training phases — was never a measurement of model quality. It is a training-loss
curve across three different, non-comparable loss functions. **Whether the model actually
generated good motion was never measured at all**, on top of training on a decode error and an
objective with a mathematically guaranteed trivial solution.

### 1.2 A rebuilt evaluation harness reproduces a published reference number

The vendored, MIT-licensed evaluator (Guo et al.'s own text-motion retrieval and FID pipeline,
the same instrument used across this field's published leaderboard) was validated against its
own published ground-truth reference (R-Precision-top3 = 0.797 ± 0.002, and 0.7977 in the
authors' own bundled evaluation log) across four independent **full-split** runs of this project,
on real motions with no model involved: 0.7969, 0.8013, 0.7950, 0.7950 — every value within
**0.0036** of 0.7977 (the largest single deviation, from the 0.8013 run). A fifth value, 0.8036,
exists in this project's own records but is a measurement on a *restricted subset* (only the
captions a later truncation-rule check happened to alter), not a full-split reproduction, and is
excluded from this claim for exactly that reason — stated here rather than left for a reader to
find it unexplained and doubt the whole claim. **This harness is trustworthy**, which is the one
property every downstream comparison in this project actually depends on.

### 1.3 Caption truncation destroys measurable text-motion alignment — a real, resolved finding

The original project's actual defect was pairing a caption's first clause with a single,
often-unrepresentative frame. Measured directly, with no model and no training involved, on real
motions: truncating a caption to its first action clause (the original project's own rule,
reproduced faithfully from its archived code) drops R-Precision-top3 from **0.8013 to 0.6563** —
a **0.145** drop, roughly **9x** the measured noise floor (batching variance of ~0.016). Where
the rule actually fires (55.9% of captions; the rest are short enough that the rule leaves them
unchanged), the conditional cost is larger: **0.27**, roughly **17x** the noise floor.

Three further controls, run because the first result invited an easy but wrong story, decompose
this cleanly:

- **Volume vs. content selection.** A content-neutral control (naive first-N-words, same length
  as the original's rule) scores statistically the same as the original's own rule (0.6468 vs.
  0.6552). **The original project's "first-action segmentation" — presented as a deliberate
  design contribution with its own POS-tagging logic — was neither clever nor uniquely harmful.
  It was one of many ways to discard about a third of a caption's words, and the cost came from
  the discarding itself, not from any skill in choosing where to cut.**
- **Position, resolved properly.** A naive single check for whether *where* the cut lands
  matters (front of the sentence vs. an arbitrary middle window) looked like a null result — but
  the randomized-window arm's own seed controls a genuinely different *treatment* each time
  (which window gets drawn), not incidental noise, so a single draw could not distinguish "no
  effect" from "one unlucky draw." Averaged over 8 independent placement draws, a real, small
  effect resolves: keeping a caption's front costs about 0.018 fewer alignment points than an
  average middle window, a genuine but modest **~7%** of the total truncation cost, significant
  at roughly **5.9 standard errors**. The remaining **~93%** of the cost is volume alone.

**The honest summary: caption truncation costs retrievable text-motion alignment in close
proportion to how much text is removed, almost independent of which part is removed.** This is a
real, three-times-checked, still-standing finding.

### 1.4 A model trained here learns real, generalizable text conditioning at a tiny budget

A from-scratch model (MDM's real published architecture, 17.9M parameters, not a toy) trained
for 3,000 steps — 0.63% of the original paper's training budget, on a materialized ~4,400-
sequence subset, disjoint from its evaluation split — scored R-Precision-top3 = **0.2969**
against a pre-registered chance threshold of 0.09375: **3.17x chance**, evaluated on held-out
captions it never trained on, so this cannot be memorization. **The pipeline genuinely learns.**

### 1.5 A non-obvious finding about the evaluation instrument itself

While building a demonstrator, a full-corpus retrieval check (does a caption's own true motion
rank first among ~8,200 real candidates, using the same validated text-motion embedding space
behind every R-Precision number in this project) collapsed to about **1%** — even though the
identical space, restricted to the field's own standard 32-candidate batches, is the instrument
already validated in §1.2. Diagnosed directly: a caption's own true motion scores 0.978 cosine
similarity against its own text embedding, yet ranks **264th out of 8,198** candidates, because
every corpus motion clusters in a narrow 0.97-0.99 similarity band at that scale. **The field's
standard text-motion evaluator has enough resolution to rank one caption against 31 distractors,
and not against thousands** — which is a real, checked reason the published R-Precision protocol
specifies a 32-candidate pool, not an arbitrary convention. This is a property of an instrument
the whole text-to-motion field shares (Guo et al.'s evaluator, used by every number on this
field's published leaderboards), found here because a retriever was measured before being
shipped rather than trusted because its underlying component was validated elsewhere.

### 1.6 A working, interactive demonstration of the finding

A local demo (`demo/`) lets anyone type a motion caption, watch the original project's own
truncation rule cut it live, and see the nearest real training motion each version of the
caption retrieves — no generation, no wait, no curated examples. On the first uncurated caption
tried during this project's own verification ("a person walks forward and then sits down on a
chair"), the full caption correctly retrieved a real motion of someone sitting down; the
truncated version retrieved a different real motion of someone only walking — the sitting
information genuinely gone. Across 300 real captions, this reproduces the same effect as §1.3
through an entirely different, non-embedding retrieval method (TF-IDF lexical similarity):
self-retrieval falls from 74.7% (full caption) to 51.7% (truncated), changing which motion gets
retrieved in 34.7% of cases — independently reproduced by a second measurement (78.3% -> 55.0%,
46.0% changed) to within a third of a percentage point on the size of the drop.

---

## 2. What was NOT established

**D-03 (the requirement that this project's evaluation harness reproduce a published *generated-
model* number, not just a ground-truth one) is UNRESOLVED.** §1.2 validates the harness against
real motions; reproducing MDM's own published FID score on its own released checkpoint did not
succeed at any sample size this hardware could afford. From this project's own measured
generation rate (~39 minutes per 128 samples on this CPU): **a single full-scale replication
(n≈1,000, matching the published protocol's sample count) would cost roughly 5 CPU-hours on this
hardware, and the full published 20-replication protocol roughly 100 CPU-hours** — the
checkpoint's own bundled evaluation log reports "about 12 Hrs" for that same 20-replication
protocol, but that figure is the original authors' hardware, not this project's; stating it
without that distinction would understate this project's own actual constraint. A reduced-scale
attempt that *was* affordable (n=128) gave an unstable, uninterpretable FID number, itself
diagnosed and explained: FID's covariance estimate is unusable at this project's affordable
sample sizes, a separate real finding in its own right (§1.5's sibling — an instrument-limits
finding about FID rather than R-Precision). **Every number in this document that involves a
trained model is internally-comparable-only — this project makes no claim of comparability to
any published leaderboard result.**

**Whether §1.3's retrieval-space truncation cost propagates into actual generated motion quality
is genuinely open, not resolved either way.** This was tested directly: two models (one trained
on full captions, one on the original's truncated captions, identical architecture and budget)
were compared, and the observed gap was **0.80 standard errors** — smaller than pure sampling
noise at this project's affordable generated-sample count (n=128 per arm). Resolving this at a
conventional significance threshold would need roughly **1,780 generated samples per arm per
seed — about 9 CPU-hours of generation alone, per arm, per seed**, computed directly from this
project's own measured statistics, not estimated. **This is a bound on what this project's
hardware can detect, not a bound on whether the effect is real** — a legitimate result about
what a laptop-scale, CPU-only rebuild can and cannot establish, not a failed experiment. It was
anticipated as a possible outcome before the comparison was run, not discovered only afterward.

**No claim is made about generation quality in general.** The model in §1.4 is real but severely
undertrained by design (0.63% of a competitive training budget); its own outputs are not fit for
comparison to any published system, and the demonstrator in §1.6 states this plainly rather than
show generated output as if it were this project's contribution.

**One dataset (HumanML3D), one architecture (MDM's transformer-encoder diffusion model), one
compute budget.** Nothing here claims to generalize to other text-to-motion datasets,
architectures, or evaluators beyond the one specific, checked instance in §1.5.

---

## 3. How to see it

Run the interactive demonstrator (`demo/README.md` for exact setup): type any caption, watch it
get truncated live by the original project's own rule, and see the nearest real motion each
version retrieves, instantly, with no curation. The measured numbers from §1.3 and §1.6 are
printed directly on the page, including the "internally-comparable-only" caveats above — not
left to a report nobody reads. `docs/METRICS_EXPLAINED.md` explains what R-Precision and FID
actually measure, for anyone who wants that; nothing in this document requires reading it first.

---

## 4. What it cost

The training and generation work behind §1.4 and the unresolved comparison in §2 together took
roughly **5 CPU-hours** on a single Apple Silicon laptop, with no GPU and no rented compute at
any point in this project. The forensic and retrieval-space findings in §1.1, §1.3, §1.5, and
§1.6 needed no model training at all — minutes of CPU time each. **That is the actual answer to
the question this project's own framing (a solo student rebuild, for educational and research
value, with zero commercial ambition) was always about: what a single person, alone, with a
laptop and no budget, can and cannot establish about a failed research project.** What could be
established: a full forensic diagnosis of why the original failed, a validated evaluation
harness, a real and three-times-checked finding about information loss in text conditioning, a
working model that demonstrably learns, a non-obvious finding about a shared field instrument,
and an honest, working demonstration of all of it. What could not: whether that finding
propagates into generation quality, at any sample size this hardware affords — and knowing
*exactly* how much larger a hardware budget would be needed to answer that question is itself
part of what this project produced, not a gap left unexamined.
