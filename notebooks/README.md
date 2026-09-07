# Notebooks — entry point

Nine executed notebooks, each demonstrating one finding this project has already earned, in the
six-layer structure (`reviews/NOTEBOOK_STYLE_GUIDE.md`): question, intuition, setup, measurement,
what it means, what would change my mind. Every number in every notebook is produced by a cell in
that notebook — none are pasted from `LEDGER.md` or `docs/`. Ordered below by argument, not by
filename, so the arc reads in one pass: representation and the F1 bug, then the evaluator's own
limits, then the conditioning finding, then the statistical discipline that governs how any of
this should be trusted, then a craft notebook on PyTorch/MPS fluency that stands apart from the
research arc above it.

**Review status, stated plainly per the reviewing session's own request:** "reviewed" means an
independent reviewing session re-derived the notebook's numbers and opened its rendered figures
before accepting it (`reviews/REVIEW_QUEUE.md`). "Self-audited only" means the producing session
(this one) re-checked its own work the same way, but no second party has done so yet — weaker
evidence than an independent review, stated here rather than left to look like one.

**Dataset provenance caveat (`reviews/REVIEW_QUEUE.md` SUP-20260907-90), stated once here rather
than buried in each notebook:** every notebook that touches HumanML3D data draws it from
`TeoGchx/HumanML3D` on HuggingFace — an individual's unlicensed, unattributed mirror, not the
official release channel (HumanML3D itself derives from AMASS, which forbids redistributing
processed data). The data's *integrity* is strongly corroborated across this project's own checks
(ground-truth R-Precision 0.797-0.812 against a published 0.797; bone lengths constant to 3.4e-07;
correct (T,263) float32 layout) — but its *provenance* is undocumented, and no notebook below
should be read as implying official acquisition.

## 1. Representation and the F1 bug

- **[03_263d_representation_and_f1_bug.ipynb](03_263d_representation_and_f1_bug.ipynb)** — *Does
  slicing HumanML3D's 263-d motion vector at `[:66]` instead of decoding it properly produce a
  visible error, or a silent one?* **Silent.** The wrong-slice decode still renders a
  recognizable-looking skeleton; only a bone-length invariant (constant length across every frame
  of a rigid skeleton) exposes it — coefficient of variation ~5 orders of magnitude higher under
  the broken decode than the correct one, cross-checked against `FORENSICS.md` F1's own original
  250-sample measurement (24.49% vs. 25.81% mean CV, same worst bone). **Reviewed**
  (`reviews/REVIEW_QUEUE.md` SUP-20260907-85/86/87, three real figure defects found and fixed).
- **[04_fid_covariance_rank_deficiency.ipynb](04_fid_covariance_rank_deficiency.ipynb)** — *Is FID
  a stable number at the sample sizes this project can afford?* **No.** 128 generated motions'
  feature covariance has numeric rank exactly 127 (n-1, the theoretical maximum) against 263
  parameters — a ~10-order-of-magnitude eigenvalue cliff. FID between the same bit-identical
  generated set and different reference constructions varies 1.577 -> 0.214 -> 0.109 as reference
  size grows 128 -> 1024 -> 2000, cross-checked against `docs/LANDMINES.md` §14's three prior FID
  values. **Reviewed** (`reviews/REVIEW_QUEUE.md` SUP-20260907-94 confirms complete).

## 2. The evaluator's own limits

- **[02_tmr_second_evaluator.ipynb](02_tmr_second_evaluator.ipynb)** — *Does a second, independent
  evaluator (TMR) agree with this project's primary one (Guo et al.)?* **Yes, once four real bugs
  were found and fixed** (wrong tokenization, a double-sort that silently swapped a handful of
  tied-length embeddings — found after two rounds of review had already closed this notebook, and
  which also corrects an earlier claim that motion embeddings were batch-composition-sensitive;
  they aren't — missing evaluator-specific normalization; see `docs/LANDMINES.md` §25) — Guo
  R-Precision-top3 lands at an exact 0.7578, matching this project's own prior target to the
  digit; TMR also at 0.7578; per-sample correlation between the two evaluators rising from -0.006
  to 0.328 across all four fixes. The notebook's *original* headline ("the evaluators are
  uncorrelated") was itself an artifact of the bugs, corrected explicitly in its own closing text
  rather than silently replaced — twice now. **Reviewed**
  (`reviews/REVIEW_QUEUE.md` SUP-20260907-88/91/92, closed).

## 3. The conditioning finding

- **[01_clip_spatial_blindness.ipynb](01_clip_spatial_blindness.ipynb)** — *Does MDM's frozen CLIP
  text encoder actually separate spatial language (left/right, forward/backward) as well as it
  separates other kinds of modifiers?* **No.** n=40 matched pairs, Cohen's d=+0.995, p<0.00001
  (Bonferroni-corrected), and 57.6% of HumanML3D's own captions contain a spatial term — not a
  narrow case. **Reviewed** (revamped directly by the reviewing session under an author-granted
  scoped override; accepted at `reviews/REVIEW_QUEUE.md` SUP-20260907-82/83/84).
- **[01b_pooling_probe.ipynb](01b_pooling_probe.ipynb)** — *Is the spatial-blindness finding an
  artifact of MDM pooling CLIP's per-token output down to one vector?* **No.** At n=40, the
  spatial-minus-control gap *widens* from pooled (+0.0265, d=+0.995) to the single
  most-divergent per-token position (+0.1106, d=+0.724) — CLIP's per-token features are already
  spatially weak; a fix that only changes how MDM reads CLIP's output would not help. **Reviewed**
  (`reviews/REVIEW_QUEUE.md` SUP-20260907-93, "a negative result that saves the project money").
- **[05_spatial_subset_split.ipynb](05_spatial_subset_split.ipynb)** — *If a future experiment
  wants to compare generation quality on spatial vs. non-spatial captions, does the obvious test
  split support that comparison cleanly?* **Not without controlling for a real confound.**
  HumanML3D's official test split gives 2,448 spatial / 1,750 non-spatial captions (58.3%,
  agreeing with notebook 01's own 57.6% corpus-wide figure) — but spatial captions are 4.18 words
  longer on average (Cohen's d=+0.592, a medium, real effect, not a large-n artifact), while a
  motion-length difference in the same split is statistically significant but negligible in size
  (d=-0.104). **Reviewed** (`reviews/REVIEW_QUEUE.md` SUP-20260907-95, exact re-derivation; its
  own follow-up on how to interpret the confound's direction is addressed in
  `docs/EXPERIMENT_DESIGN_E2.md` §3, not in this notebook).
- **[06_cfg_training_loss_collapse.ipynb](06_cfg_training_loss_collapse.ipynb)** — *The archived
  (pre-reboot) project folded classifier-free guidance into its training loss instead of applying
  it only at inference time. Does that give a text-ignoring model zero gradient pressure to start
  using the caption, as `docs/LANDMINES.md` §11 originally argued?* **No — the opposite.** A toy
  system built to reproduce the exact scenario shows the gradient on the conditioning pathway is
  not zero; it amplifies *linearly* with the guidance scale, and causes training divergence at
  moderate learning rates instead of a silent standstill — matching §11's own separate point about
  gradient instability (the `5*tanh` squash and gradient clipping as compensating hacks) rather
  than its "no gradient pressure" framing. `docs/LANDMINES.md` §24 records this as a correction to
  §11, not a retraction. A second finding, added after independent review (below): the archived
  formula's own training loss converges *lower* than the correct formula's, not because it
  discards conditioning (checked directly — it doesn't), but because its loss never scores the
  no-caption case in isolation the way the correct formula must — `CLAUDE.md`'s "a training loss
  is not a result," demonstrated on the exact bug this project's predecessor died of. **Reviewed**
  (`reviews/REVIEW_QUEUE.md` SUP-20260907-98 independently re-derived both headline results and
  requested the loss-decomposition addition above; a self-audit performed 2026-09-07, before that
  review landed, had already opened every rendered figure and found and fixed one real defect — a
  clipped subplot title).

## 4. The statistical discipline

- **[07_why_six_samples_is_not_a_finding.ipynb](07_why_six_samples_is_not_a_finding.ipynb)** —
  *E1's headline comparison found a gap of six samples out of 128 (38 vs. 44). Is that a result?*
  **No, and this notebook shows two independent reasons why.** The binomial floor alone puts the
  gap at 0.80σ — well short of any normal significance threshold. More directly: re-running the
  same arm with nothing changed but the random seed (seed 10 vs. seed 20) reproduces the *exact
  same raw counts* (38 and 44) from randomness alone. The notebook also shows the experiment was
  not simply "underpowered" — it was powered against the wrong effect size (its own 0.80σ noise
  reading, which is circular by construction, rather than the pilot's own hypothesis-motivated
  effect, which needed an affordable n). This is the direct precedent for
  `docs/EXPERIMENT_DESIGN_E2.md` stating its own possible underpowering up front rather than
  discovering it after the fact. **Reviewed** (`reviews/REVIEW_QUEUE.md` SUP-20260907-98
  independently re-derived the binomial statistics and MDE table, matching exactly aside from a
  disclosed rounding-convention difference; a self-audit performed 2026-09-07, before that review
  landed, had already re-derived every headline number in a separate scratch script against the
  raw artifact JSONs directly, and found and fixed one real figure defect
  and fixed — two overlapping annotation text boxes).

## 5. Craft and fluency

- **[08_pytorch_mps_silent_failures.ipynb](08_pytorch_mps_silent_failures.ipynb)** — *What does
  PyTorch (and Apple Silicon's MPS backend specifically) get silently wrong, if you don't know to
  check?* A craft notebook, not a motion-generation one, built at the author's request: nine real
  specimens (a `(N,)` vs. `(N,1)` broadcasting bug that trains anyway; a device-selection function
  that silently falls back to CPU; the identical cast-after-transfer bug independently present in
  a second vendored file; a double-sort that silently swaps tied-length embeddings — found while
  building this notebook, and the direct cause of notebook 2's fourth-bug correction above; a
  normalization mismatch that flips a published-looking conclusion; a tokenization mismatch that
  halved a headline metric with zero exceptions raised; RNG state silently consumed by an
  unrelated diagnostic; `torch.tensor()`'s quiet copy-and-detach; and gradient accumulation from a
  missing `zero_grad()`), each demonstrated wrong-way-next-to-right-way with real numbers, plus
  three candidates tested directly and dropped when they didn't reproduce, stated as such. Closes
  with five reusable habits — including this project's own invented batch-invariance check — that
  cover every specimen in the catalogue. **Self-audited only — not yet independently reviewed**
  (built and self-checked 2026-09-07: every printed claim backed by a runnable cell, one
  self-referential false positive in its own verification code found and fixed before finalizing).

## A note on the self-audits above

A self-audit is the producing session checking its own work with the same method a reviewer would
use — opening rendered figures rather than trusting the code that made them, re-deriving numbers
independently rather than trusting the notebook's own cells, cross-checking against an existing
project number rather than accepting a result in isolation. It is real verification, and it found
two real defects (both fixed, both described above). It is still not the same as an independent
second party checking the same claims — that is why notebooks 06 and 07 are marked "self-audited
only" rather than "reviewed," and why that distinction is stated here rather than left implicit.
