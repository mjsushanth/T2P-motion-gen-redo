# T2P-Reboot — text-to-motion generation, built around measurement

Given a sentence like *"a person walks forward, then turns left,"* generate a 3D human motion
sequence that matches it. This project is a ground-up rebuild of that problem, with one rule
above all others: **no claim of quality without a number to back it, and no number without the
code that produced it sitting right next to the claim.**

That discipline is the actual product here, as much as any model. It shows up most clearly in
the notebooks below.

---

## The notebooks — the core of this project

Every notebook answers exactly one question, end to end: the question, the intuition behind it,
the measurement, and what it means — with every number computed live in the notebook itself,
never pasted in from a prior run. Eight questions asked and answered so far.

<p align="center">
  <img src="notebooks/03_skeleton_side_by_side.png" alt="A subtly wrong pose decode looks fine as a tangle of lines, and unmistakably wrong once rendered as a skeleton" width="720">
</p>

1. **[Does a subtly wrong way of reading 3D pose data look fine, or does it show?](notebooks/03_263d_representation_and_f1_bug.ipynb)**
   It looks completely fine — until you check that a bone's length stays constant across frames.
   The wrong decode fails that check by five orders of magnitude.

2. **[Is a standard motion-quality score (FID) trustworthy at small sample sizes?](notebooks/04_fid_covariance_rank_deficiency.ipynb)**
   No. The same set of generated motions produces a score that varies several-fold depending on
   how large the comparison set is — a fact about the metric, not about the motion.

3. **[Does a second, independent evaluator agree with the primary one?](notebooks/02_tmr_second_evaluator.ipynb)**
   Eventually, yes — but only after three real bugs in how the two were being compared were
   found and fixed. Trusting a single evaluator without a cross-check would have shipped a wrong
   conclusion.

4. **[Does the text encoder actually understand spatial language — "left," "right," "behind"?](notebooks/01_clip_spatial_blindness.ipynb)**
   No. CLIP measurably struggles to tell spatial opposites apart, and spatial words appear in the
   majority of real captions in this dataset — not a rare edge case.

5. **[Is that blind spot caused by how the model reads CLIP's output, or is it inside CLIP itself?](notebooks/01b_pooling_probe.ipynb)**
   Inside CLIP itself. A wiring fix on the reading side would not fix it.

6. **[Before comparing "spatial" vs. "non-spatial" captions, is that comparison even fair?](notebooks/05_spatial_subset_split.ipynb)**
   Not without care — the two groups differ in caption length in a way that could easily be
   mistaken for a spatial-language effect if nobody checked first.

7. **[Could one specific training bug have quietly taught a model to ignore text completely?](notebooks/06_cfg_training_loss_collapse.ipynb)**
   Yes — demonstrated directly on a rebuilt version of the bug. And its training loss looks
   *better*, not worse, while this happens: exactly why a loss curve alone is never a result.

8. **[When two results differ by a handful of samples, is that a real effect?](notebooks/07_why_six_samples_is_not_a_finding.ipynb)**
   No — rerunning the identical setup with only the random seed changed reproduces the same
   "effect" out of pure chance.

<p align="center">
  <img src="notebooks/07_circularity_trap.png" alt="Required sample size explodes as the assumed effect size shrinks — the smaller the effect you assume, the more data you need to detect it" width="640">
</p>

Full index with figures and status: **[notebooks/README.md](notebooks/README.md)**.

---

## The research record

Alongside the notebooks, this project keeps a written trail of *why*, not just *what* —
searchable, dated, and meant to be read by whoever picks this up next:

- **`docs/LANDMINES.md`** — a running catalogue of traps that produce a plausible wrong answer
  with no error message, each one caught and documented so it isn't repeated.
- **`docs/DECISIONS.md`** — every non-obvious design choice, with the reasoning behind it.
- **`docs/EXPERIMENT_LOG.md`** — every experiment run, including the ones that resolved to "not
  enough evidence either way" rather than a convenient story.
- **`docs/GLOSSARY.md`** — every term and acronym, defined once, the first time it appears.
- **`docs/EXPERIMENT_DESIGN_E2.md`** — the next experiment, fully specified and pre-registered
  before any of it runs: fixing CLIP's spatial blindness with a small adapter, tested against a
  built-in control for making anything else worse.

---

## Status

| Question | Where it stands |
|---|---|
| Can this project trust its own scoring? | Yes — validated against a second, independent evaluator |
| Is the text encoder's spatial-language weakness real? | Yes — measured, and confirmed not to be a wiring artifact |
| Is the one training bug that could have doomed the original approach understood? | Yes — reproduced and explained directly, not just described |
| Does the evaluation split hold up under scrutiny? | Checked for hidden bias before being used for anything |
| Is a fix for the spatial-language weakness designed? | Yes — pre-registered, not yet run |
| Has a full training run happened? | Not yet — deliberately, until the above was settled |

---

## Going deeper

- **`docs/00_START_HERE.md`** — five-minute orientation for anyone new to this repo.
- **`REBUILD_SPEC.md`** — the architecture and experiment plan.
- **`LANDSCAPE.md`** / **`POSITIONING.md`** — where this sits relative to published work.
- **`LEDGER.md`** — the complete, append-only log of every unit of work behind all of the above.

## Hardware

Apple Silicon. Runs on MPS, not CUDA.
