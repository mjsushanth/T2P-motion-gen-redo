# EXPERIMENT_DESIGN_E2 — pre-registered design for a spatial-conditioning repair experiment

**Status: DESIGN ONLY. Nothing in this document has been run.** Written before any budget is
spent, per this project's own standing rule (E1 died partly because its design was written after
its budget was assumed — `docs/DECISIONS.md` D-23, `docs/LANDMINES.md` §18). Author-directed via
the reviewing session (2026-09-07), no sign-off required to write this; sign-off *is* required
before any arm below actually runs.

**The evidence this design rests on, all already established, none re-asserted here without a
pointer:**
- `notebooks/01_clip_spatial_blindness.ipynb`: MDM's frozen, pooled CLIP text embedding
  under-separates spatial minimal pairs (left/right, forward/backward, ...) relative to
  non-spatial modifier pairs of the same edit distance — n=40/group, Cohen's d=+0.995, p<0.00001
  (Bonferroni-corrected). 56.9-57.6% of HumanML3D's own captions contain a spatial term — not a
  narrow case.
- `notebooks/01b_pooling_probe.ipynb`: the deficit is **not** created by MDM's pooling step. It
  is present, and *widens*, in CLIP's own per-token representations (gap +0.0265 pooled ->
  +0.1106 at the most-divergent token, both far past Bonferroni correction). **A fix that only
  changes how MDM reads CLIP's output (e.g. "use per-token features instead of pooling") is
  therefore not expected to help** — the encoder itself needs to change, not just how it is read.
- `notebooks/05_spatial_subset_split.ipynb`: HumanML3D's official test split partitions into
  2,448 spatial / 1,750 non-spatial captions (58.3%, agreeing with notebook 01's corpus-wide
  57.6%). **A real confound was found**: spatial captions are 4.18 words longer on average
  (Cohen's d=+0.592, p<0.00001) — a medium effect, not a large-n artifact. Motion length also
  differs but negligibly (d=-0.104). **Any comparison between these two subsets must control for
  caption length or it is not measuring what it claims to measure.**

## 1. The question

**Does repairing CLIP's spatial-language separation improve MDM's generation specifically on
spatially-worded captions, without degrading performance on everything else?**

This is not "train a better text-to-motion model." It is a narrow, falsifiable question about one
specific, already-measured deficit, on a pretrained checkpoint, with a small modification — matching
the strategic direction already settled (pretrained checkpoint + modification, never from scratch;
`LEDGER.md` Item 67's own note on the relayed strategic context).

## 2. Arms

**Baseline — unmodified.** The released MDM checkpoint
(`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/model000475000.pt`), exactly as this
project has used it throughout E0b/notebooks 02/03/04. No training. This is not a new experiment —
its R-Precision numbers already exist (E0b: 0.7578 aggregate, notebook 02: recomputed and
cross-checked) — what is new is scoring it **separately on the spatial and non-spatial subsets**
(§3), which nobody has done yet.

**Treatment — baseline + a small spatial adapter, diffusion model otherwise frozen.**

- **What the adapter is.** A single trainable linear layer (or a 2-layer MLP with a residual
  connection to the identity — the simpler linear version should be tried first, per this
  project's own "surgical, not clever" preference; escalate to the MLP only if the linear version
  shows a floor effect), operating on CLIP's pooled 512-d output.
- **Where it sits.** Between CLIP's frozen `text_projection` output and MDM's own
  `embed_text` linear layer (`model/mdm.py`, `self.embed_text = nn.Linear(self.clip_dim,
  self.latent_dim)`). The adapter transforms the 512-d pooled CLIP vector into a new 512-d
  vector, which then feeds into `embed_text` and the rest of the (frozen) diffusion model
  completely unchanged. CLIP itself stays frozen (as MDM already keeps it, `load_and_freeze_clip`)
  and the diffusion transformer, `embed_text`, and every other MDM weight stay frozen too — **only
  the new adapter is trained.**
- **Why this specific location, not fine-tuning CLIP itself.** MoCLIP (arXiv:2505.10810)
  fine-tuned CLIP for motion contrastively and reports only "competitive" FID afterward — the
  wording authors use when a change did not help and did not obviously hurt enough to bury the
  number. **That is the exact failure mode this design exists to avoid repeating**, and the reason
  the adapter sits after CLIP (frozen) rather than fine-tuning CLIP's own weights: a small,
  isolated adapter is far less able to silently degrade everything else than fine-tuning the whole
  encoder is, and if it does degrade something, the blast radius is one linear layer, not CLIP's
  entire representation space.
- **Training objective.** A contrastive loss on a **synthetic set of spatial minimal pairs**,
  templated (not the same 40 pairs used for evaluation in notebooks 01/01b — training on the
  exact evaluation pairs would be circular). E.g. combinatorially expand a set of sentence
  templates (`"a person {verb}s their {spatial} {noun}"`, `"a person moves {spatial}"`, etc.)
  crossed with the spatial-term list already established (`left/right/forward/backward/
  clockwise/counterclockwise/in front of/behind`), targeting a few hundred to a few thousand
  synthetic pairs — cheap to generate, no motion data needed for this step at all. Loss: pull
  the adapter's output for a minimal pair *apart* by a fixed margin relative to CLIP's own
  pooled-space distance, while a regularization term keeps the adapter close to the identity
  transform on a held-out sample of ordinary (non-minimal-pair) captions, specifically to protect
  non-spatial performance (the regression control, §4).

**A third, deliberately-not-run arm named for completeness:** adapter + light fine-tuning of the
diffusion transformer itself (unfreezing a few of MDM's own layers). **Not part of this
pre-registration.** If the frozen-adapter arm shows a null result, the live hypothesis this would
test is "the deficit needs to reach the diffusion model's own weights, not just its input
embedding" — but that arm costs meaningfully more compute and carries a real risk of the same
MoCLIP-style regression, and should be designed and reviewed on its own if the cheaper arm's
result actually motivates it, not bundled into this pre-registration speculatively.

## 3. The split that makes this novel

Report R-Precision-top3 on the spatial subset and the non-spatial subset **separately**, for both
arms — four numbers, not two:

| | spatial subset (n=2,448 available) | non-spatial subset (n=1,750 available) |
|---|---|---|
| baseline | R-Precision-top3, spatial | R-Precision-top3, non-spatial |
| treatment (adapter) | R-Precision-top3, spatial | R-Precision-top3, non-spatial |

**Nobody in the literature reports this split** (`guidance/RESEARCH_E_novel_directions.md`
confirms it) — every published text-to-motion evaluation reports one aggregate R-Precision
number. Reporting the spatial and non-spatial subsets separately, for the same model, is the
actual contribution of this design, independent of which way the treatment arm's result lands.

**The confound control, made concrete (per notebook 05's own finding, not just acknowledged):**
spatial captions are 4.18 words longer on average. To avoid a caption-length effect masquerading
as a spatial-conditioning effect, the comparison is **length-stratified**, not pooled: within each
subset, split into short/medium/long caption-length terciles (computed once, from the full test
set's own word-count distribution) and report R-Precision per tercile per subset per arm.
**A treatment effect that only appears in the length tercile where the two subsets differ most is
a warning sign, not a confirmation** — a real spatial-conditioning effect should show up across
length terciles, not concentrate in the one where the confound is strongest.

## 4. The regression control

**Non-spatial R-Precision must not measurably degrade.** This is not a secondary check — it is a
pre-registered arm of the result, with its own explicit failure condition (§6). The adapter's own
training objective (§2) includes a regularization term specifically to protect this, but the
regularization being *present in the design* does not mean it *works* — that is an empirical
question this experiment answers, not one it assumes the answer to.

## 5. Power calculation

**Effect size to detect: unknown, by construction — this is the actual gap in this project's
knowledge that a power calculation has to be honest about**, not a number carried over from
notebooks 01/01b (those measure CLIP *embedding* separation, cosine similarity on hand-written
minimal pairs — not R-Precision on real generated motions, which is two representational steps
downstream and not guaranteed to move by a comparable amount, or at all).

**Binomial floor, using the same formula this project already validated in D-28/`docs/LANDMINES.md`
§18** (`n = z² · 2p(1-p) / delta²`, p≈0.35 — this project's own severely-undertrained-checkpoint
R-Precision-top3 baseline, per E0b/notebook 02, not the published-frontier 0.76 a fully-trained
checkpoint would use):

| delta (R-Precision-top3 points) | n/arm needed, 3σ | n/arm needed, 2σ |
|---:|---:|---:|
| 0.15 | 156 | 69 |
| 0.10 | 350 | 156 |
| 0.05 | 1,401 | 623 |
| 0.03 | 3,891 | 1,730 |

Against the subset sizes notebook 05 established (2,448 spatial / 1,750 non-spatial available in
the *full* test set — not all of it necessarily affordable to generate), **this design can afford
a 3σ-detectable effect down to roughly 0.10-0.12 R-Precision points on the spatial subset without
exhausting it, and down to roughly 0.12-0.14 on the smaller non-spatial subset**, if every
available caption in each subset were generated once. Realistic affordable n (§7) is smaller than
the full subset, which pushes the actually-detectable effect size up from this floor — stated
plainly in §7, not hidden in this table.

**If the honest MDE at the affordable n turns out larger than any effect worth finding, that is a
legitimate, useful conclusion — the one E1 reached too late (`docs/DECISIONS.md` D-26/D-28) — and
this design is pre-committing to say so plainly if it happens, not to quietly lower the bar.**

## 6. Pre-registered failure / abandonment conditions

Stated before any run, so a disappointing result cannot be re-narrated afterward:

1. **Non-spatial R-Precision drops by more than the seed-to-seed spread already characterized for
   this checkpoint** (D-28: same-arm seed variance alone can equal 0.047 R-Precision-top3 points
   at n=128) — if the treatment arm's non-spatial score drops by more than this floor, **abandon
   the adapter design regardless of what happens on the spatial subset.** A model that trades
   non-spatial competence for spatial competence has not fixed anything.
2. **The spatial-subset improvement, if any, is smaller than the pre-registered MDE at the actual
   affordable n (§7).** A numerically-higher spatial R-Precision that sits inside the binomial
   noise floor is not a finding — report it as indistinguishable from noise, exactly as D-26/D-28
   already established this project will do rather than over-read a favorable-looking number.
3. **The spatial-subset improvement is concentrated in one caption-length tercile that also
   happens to be where the length confound (§3) is strongest.** This is the specific signature of
   a confounded result and is treated as inconclusive, not as evidence for the adapter, until a
   length-matched re-analysis can rule the confound out.
4. **The adapter's training loss does not converge, or converges to near-identity** (i.e. the
   regularization term dominates and the adapter learns to do nothing) — report this as "the
   contrastive objective as specified did not train," a design failure worth recording in
   `docs/LANDMINES.md`, not silently re-tuned until something moves.

## 7. Compute plan

**Free tier: Kaggle, ~30 GPU-hours/week, P100 (16GB) or T4x2 (32GB), 9-hour session cap, no
credit card** (per the relayed strategic context, `LEDGER.md` Item 67 — not independently
re-verified by this session; treat as UNVERIFIED until this project itself confirms it directly
against Kaggle's own current documentation before relying on it for a real run).

**This project has never measured a GPU rate directly — every number below is a translation from
this project's own Mac CPU/MPS measurements (D-27/D-28), not a fresh measurement, and is labeled
as such.**

- **Adapter training (the cheap part).** A single linear layer, a few thousand synthetic pairs,
  a contrastive loss with no motion data involved at all — this is a small classification-scale
  training job, plausibly under an hour on any GPU tier Kaggle offers, including the 2016-era
  P100. **Unmeasured; should be timed directly in the first few minutes of the first Kaggle
  session, not assumed**, before committing to a specific n of synthetic pairs.
- **Generation (the expensive part, dominates cost per D-28's own finding for this exact
  architecture).** This project's own measured MPS rate is 3.15 s/sample (D-27/D-28, batch 32,
  1000 diffusion steps, `trans_enc` 17.9M-param architecture — the same architecture this design
  reuses unchanged). A P100/T4 GPU should be substantially faster than Apple Silicon MPS for this
  workload, but **by how much is not measured by this project and should not be assumed** — the
  relayed strategic context's own caveat (a P100 is 2016 Pascal with no bf16, so an RTX-5090-based
  hour estimate needs inflating 2-4x) cuts the other way for a *CPU-based* estimate translated
  *up* to a GPU: treat any pre-run estimate here as order-of-magnitude only.
  - At this project's own MPS rate (3.15 s/sample) with no GPU speedup assumed at all (the most
    conservative floor): generating the full spatial subset (2,448 samples) would cost ~2.1
    CPU/MPS-hours; the non-spatial subset (1,750) ~1.5 hours; both arms (baseline + treatment)
    roughly double this, ~7.3 hours total generation, comfortably inside one week's 30-hour
    Kaggle allocation even at zero GPU speedup, and likely a real overestimate given P100/T4 GPUs
    should outperform CPU-bound MPS meaningfully for this workload — measure this on the first
    Kaggle session before finalizing which n to actually run.
  - The 9-hour session cap constrains a single run's *length*, not the total weekly budget —
    if a full-subset generation run exceeds 9 hours, it must be checkpointed and resumed across
    sessions (a real engineering requirement, not yet designed — needed regardless of whether
    this specific experiment ever runs, since any future full-budget training run has the same
    requirement, per the relayed strategic context's own note).
- **Total estimated compute for this design as specified: comfortably inside one week's free
  Kaggle allocation, likely with room to spare** — but every number above is a translation from
  Mac-measured rates to unmeasured GPU hardware, stated as an estimate, not a measurement, and the
  first real Kaggle session should measure the actual GPU rate before any full-subset run is
  committed to.

## 8. What this design does not claim

- It does not claim the adapter will work. Section 6 exists specifically so a null or negative
  result is a complete, reportable outcome, not an unplanned-for failure.
- It does not claim spatial-language conditioning is the *only*, or even the *largest*, gap in
  this project's own generation quality — only that it is one specific, already-measured,
  cheaply-testable deficit worth checking before any larger, more expensive intervention.
- It does not authorize itself to run. This document is the design; running any arm of it still
  requires the same sign-off every other training run in this project requires, per
  `CLAUDE.md`'s own standing authorization scope.

**Would reverse if:** the power calculation (§5), once affordable n is actually measured on
Kaggle (§7), shows an MDE larger than any effect size worth caring about — in which case this
design should be shelved with that conclusion stated plainly, the same way D-26/D-28 shelved E1B's
generation-side comparison, rather than run anyway "to see."
