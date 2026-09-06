# REBUILD SPEC — Stage 2 Part B

Argues D-11, D-12, D-13 from `LANDSCAPE.md`'s evidence, plus the task-framing question the
director session explicitly reopened ("propose a different problem if the evidence points
there"). Per `docs/DECISIONS.md` D-17, this is a **redo**, not a repair: D-02/D-03
(evaluation-first) stand as engineering discipline, but the task itself is open to replacement.

D-11/D-12/D-13 are argued here but **left PENDING in `docs/DECISIONS.md`** until the director
session has read this reasoning, per our agreed process.

---

## 0. The task-framing question, first — because everything else depends on the answer

The original task was: text -> single static 3D pose, on HumanML3D (a *motion-sequence*
dataset), using frame 0 as the "pose." F3 (VERIFIED, moderate effect) showed this measurably
collapses toward a near-neutral pose regardless of caption verb. `docs/DECISIONS.md` D-12
originally framed the open question as "which static-pose dataset" (corrected HumanML3D vs.
PoseScript). Having done the landscape research, **I think that framing is itself still too
narrow, and I'm proposing a different primary task.**

### The case for switching the primary task to full text-to-motion (sequences), not static pose

1. **It eliminates F3 by construction, not by mitigation.** A caption describing "a person walks
   forward" paired with the *whole walking sequence* has no frame-selection mismatch to have in
   the first place. Every mitigation for F3 (representative-frame selection, PoseScript) is a
   patch; changing the task removes the defect's precondition entirely.
2. **The evaluation harness already exists, is MIT-licensed, and is what the whole field uses.**
   `EricGuo5513/text-to-motion` — VERIFIED MIT via `GET /repos/EricGuo5513/text-to-motion/license`
   — is the actual evaluator behind every number in `LANDSCAPE.md` §1.3. D-03's gate ("reproduce
   a published number to a stated tolerance") is concretely achievable: fetch this evaluator,
   run it against MDM's or T2M-GPT's own released checkpoint, confirm the reported FID matches.
   That is a checkable, scoped, days-not-months task. **The static-pose alternative has no
   comparably mature, fetchable, MIT-licensed harness** — PoseScript's own metrics (FID on its
   own retrieval features, mRecall) exist but are far less battle-tested (one paper's own eval
   code, not an entire field's shared instrument), and ChatPose — the most direct LLM-native
   comparison point — has no released checkpoint to validate against at all.
3. **A clear, honest difficulty ladder already exists to target.** MDM's FID (0.544, ICLR 2023)
   is a realistic first rung for a laptop-MPS rebuild; MoMask's FID (0.045, CVPR 2024) is the
   field's current ceiling. A rebuilt baseline landing anywhere in this range is a real,
   comparable number — something this project has never had.
4. **F1's fix transfers directly and cleanly.** The corrected decode (`recover_from_ric`) and the
   representation (263-d vector) are unchanged from what this project has already vendored,
   verified, and license-cleared. Nothing about the F1 fix is task-specific to static pose.
5. **It does not waste the static-pose forensics.** F3's finding stays fully relevant as a
   **controlled ablation rung** (§6 below) rather than the abandoned premise: reproduce a
   corrected-decode, frame-0-only baseline *inside the new pipeline* and measure exactly how much
   of the gap to a full-sequence model is attributable to task framing versus everything else.
   That is a real, citable number this project can produce that nobody else has, because nobody
   else made this exact mistake to measure against.

### What this does NOT rule out

- **PoseScript stays a live secondary track**, not abandoned — see §2. If time and licence terms
  allow, running the corrected pipeline's text encoder / conditioning approach against PoseScript
  too would let the project claim something about static-pose specifically, on a dataset actually
  built for it. But it is not the primary bet, for the reasons above.
- **Pose-as-control-signal (ControlNet+OpenPose) becomes the demonstrability track**, addressed
  fully in `POSITIONING.md` — it's a different kind of deliverable (a demo, not a metric) and
  doesn't compete with the research-track decision here.

**If the director disagrees with this reframing**, the fallback is D-12 as originally scoped
(PoseScript vs. corrected HumanML3D, both static-pose) — nothing below (§1, §3-§8) is
irreversible; the ablation ladder in §6 would just gain a static-pose rung 0 instead of losing
one.

---

## 1. D-12 — Dataset: corrected HumanML3D, full sequences (not frame 0)

**Recommendation: corrected HumanML3D (train/val/test splits as released), decoded via
`recover_from_ric`, full sequence, not a single frame.**

**Argued from:**
- Licence chain, VERIFIED by this session directly: HumanML3D's own upstream README states
  "Due to the distribution policy of AMASS dataset, we are not allowed to distribute the data
  directly," and its skeleton "follows the SMPL skeleton structure" (`primary_source/README.md`
  lines 37, 103). **This means the HumanML3D motion data itself — not just the processing code —
  inherits AMASS/SMPL's non-commercial restriction.** The processing code (`paramUtil.py`,
  `motion_representation.ipynb`, `common/skeleton.py`, `common/quaternion.py`) is separately
  MIT-licensed and is what this project has actually vendored (`primary_source/LICENSE`).
  **PoseScript inherits the identical restriction** — **CONFIRMED, not just likely** (peer
  review, `reviews/REVIEW_QUEUE.md` SUP-20260906-03, verified directly against
  `github.com/naver/posescript`): PoseScript's poses are **SMPL+H G format**, i.e. genuine SMPL
  body-model parameters, on top of its own CC BY-NC-SA 4.0. **There is no dataset choice
  available here that clears SMPL/AMASS's non-commercial terms.** This is not a reason to prefer
  one dataset over the other on licence grounds (they are both restricted) — it is a
  `POSITIONING.md`-level fact about what "business artifact" can mean at all under these terms,
  covered there as the headline consequence, not a footnote.
- The HF mirror already in use (`TeoGchx/HumanML3D`) was already streamed for Stage 1's F1/F3
  tests, so continuing with it (via streaming, not a bulk download) has no new data-access cost.
- Evaluator + published-number reproducibility argument in §0.

**Rejected:** PoseScript as primary (§0 point 2 — no comparably mature harness to validate
against; the licence chain does not distinguish the two, since both are now confirmed
restricted). Frame-0-only static pose from HumanML3D (the original's framing) — retained only as
an ablation rung, §6.

**Would reverse if:** the evaluator-reproduction gate (D-03) turns out not achievable with the
Guo et al. evaluator for a reason not yet known (e.g. the released checkpoint is unavailable or
corrupted) — in which case PoseScript's own smaller-but-native metrics become the more pragmatic
D-03 target.

---

## 2. PoseScript — secondary track, licence terms now fully known

If pursued at all (not gating Stage 3): CC BY-NC-SA 4.0, non-commercial + share-alike (VERIFIED,
`LANDSCAPE.md` §2.1), **plus** the SMPL non-commercial chain confirmed in §1 above (its poses are
SMPL+H G format). Code and data currently downloadable. Its evaluation metrics (FID on retrieval
features, ELBO, mRecall) are paper-specific, not a shared field instrument — weaker D-03
candidate than the HumanML3D/Guo evaluator pairing. **Before this project depends on it:**
complete SMPL/SMPL-X's own registration (§3) — PoseScript's own licence does not substitute for
that, the two stack.

---

## 3. D-11 — Representation: predict the redundant vector (matches the field); rotation+FK is an ablation, not the baseline

**Correction to this document's own starting premise:** `docs/DECISIONS.md` D-11 (as written
before this research) and `docs/LANDMINES.md` §8 both asserted that predicting joint rotations
plus a differentiable FK layer "is what MDM and MotionDiffuse do." **Checked directly against
both papers — this is not accurate**, and `LANDMINES.md` §8 has been corrected in place. MDM's
paper: "MDM can accept motion represented by either locations, rotations, or both," and for its
HumanML3D experiments specifically uses "the same representation" as Guo et al.'s own vector —
positions, velocities, *and* rotations, together, predicted directly. MotionDiffuse: its pose
state "generally contains joint rotation, joint position, joint velocity, and foot contact
conditions," again the redundant vector. **Neither flagship model's published HumanML3D number
comes from a rotation-only-plus-FK parameterisation.**

**Recommendation, revised accordingly:**

- **Baseline (what the D-03 gate targets): predict the full redundant 263-d vector**, exactly as
  HumanML3D encodes it and exactly as MDM/MotionDiffuse do. This is the representation this
  project has already vendored, decoded, and verified (F1). It is also the only choice that lets
  the D-03 gate be a like-for-like comparison against a published FID number, since that's the
  representation the published numbers were computed on.
- **Ablation rung, not the primary bet: rotation-only (root params + `rot_data`, 126-d) +
  differentiable FK via `primary_source/skeleton.py`'s already-vendored, MIT-licensed
  `forward_kinematics_cont6d`.** This is where the bone-length-invariant finding (F1's corollary:
  bone lengths are an exact dataset constant) becomes a genuine, testable hypothesis: *does
  making bone-length correctness structural (rather than learned from a redundant, internally
  self-consistent-or-not vector) measurably improve FID or reduce a directly-measured bone-length
  error metric, relative to the redundant-vector baseline?* This is a real research question this
  project's own forensics raised, not a borrowed field claim — argue it on those grounds in
  `EXPERIMENT_LOG.md`, not on "the field does this."

**Rejected:** position-output-plus-bone-length-loss (the predecessor's design) — LANDMINES §8's
underlying logic stands regardless of the MDM/MotionDiffuse correction: a loss term fights the
data term and still admits invalid poses, whereas either the redundant-vector baseline (which at
least has the *option* of a consistency check at eval time) or the FK ablation (which has no
loss term at all) both dominate it. Pure rotation-only as the *baseline* (deviates from the
reproducible-published-number strategy in D-03).

**Would reverse if:** the FK ablation turns out to beat the redundant-vector baseline decisively
in Stage 4 — in which case it gets promoted to primary and the ablation ladder's labeling
inverts. This is exactly the kind of result the ladder is designed to surface.

---

## 4. D-13 — Text encoder: CLIP token-level as baseline, DistilBERT as the validated alternative

**Recommendation:**

- **Baseline: CLIP's per-token (sequence) hidden states via cross-attention, not the pooled
  512-d vector.** Cheapest change from the original (reuses the same frozen CLIP checkpoint, no
  new encoder to source or licence-check), and directly targets the specific, sourced failure
  mode: `LANDSCAPE.md` §5.1's "text encoders bottleneck compositionality" finding is about the
  **pooled single-vector bottleneck specifically** — switching to token-level conditioning is the
  minimal fix for exactly that finding, not a bigger architectural swing than the evidence calls
  for.
- **Ablation: DistilBERT** (word-level), following TEMOS — the one HumanML3D-adjacent model with
  a directly comparable precedent for a non-CLIP encoder (`LANDSCAPE.md` §5.2 item 3; its
  specific reported numbers there are flagged UNVERIFIED and should be re-checked before citing
  in a paper, but the *architecture choice* itself is a reasonable, low-risk comparison point
  regardless).
- **Explicitly rejected for this project's scale: swapping to T5.** `LANDSCAPE.md` §5.2 item 2
  found *domain-specific* counter-evidence — a HumanML3D-scale motion model that tried replacing
  CLIP with T5 "did not observe an obvious gain," with the dataset's scale suspected as the
  limiting factor. Imagen's own preference for T5-XXL over CLIP was measured on DrawBench, an
  image-generation compositional benchmark at a vastly different data/model scale — not evidence
  transferable to a HumanML3D-scale motion model. Adopting T5 here would be paying a large
  compute/complexity cost against direct evidence that it doesn't help at this scale.
- **Mandatory regardless of choice:** a laterality-specific metric (`LANDMINES.md` §6), so the
  known left/right failure mode stays visible rather than silently absorbed into an aggregate FID.

**Rejected:** the original's frozen CLIP ViT-B/32 pooled embedding, unchanged (the specific,
sourced failure mode this is meant to fix). Full T5 swap (argued above). Sentence-transformer
generic embeddings — architecturally plausible but with no source found evaluating them on
spatial/lateral tasks specifically (`LANDSCAPE.md` §5.2 item 4); not rejected outright, just not
strong enough evidence to prioritize over DistilBERT's existing precedent.

**Would reverse if:** the laterality-specific metric shows token-level CLIP still fails badly on
left/right at this project's scale — then DistilBERT gets promoted from ablation to primary, or
explicit mirror augmentation (flip pose, swap left/right in caption) gets added as a data-level
fix regardless of encoder choice.

---

## 5. What to vendor, fork, or write from scratch — with licence notes for each

| Component | Decision | Licence status |
|---|---|---|
| HumanML3D decode/FK (`paramUtil.py`, `motion_representation.ipynb`, `common/skeleton.py`, `common/quaternion.py`) | **Already vendored**, `primary_source/` | VERIFIED MIT, `primary_source/LICENSE` added |
| Guo et al. evaluator (`EricGuo5513/text-to-motion`) | **Vendor** — this is D-03's actual gate instrument | VERIFIED MIT (`GET /repos/EricGuo5513/text-to-motion/license`, this session, 2026-09-06) |
| MDM reference implementation (`GuyTevet/motion-diffusion-model`) | **Reference only** (read for the D-03 checkpoint/eval procedure); do not fork into `src/t2p/` — our denoiser is our own architecture | VERIFIED MIT (same check, this session) |
| HumanML3D raw motion data (via HF `TeoGchx/HumanML3D` or the original AMASS pipeline) | **Stream, do not bulk-redistribute** | Inherits AMASS/SMPL non-commercial terms (§1) — this project only streams for local computation, does not redistribute the data itself, which is the actual restriction (per HumanML3D's own README, the restriction is on *distributing the data*, not on downstream research use) |
| PoseScript (if pursued, §2) | **Not yet** — complete SMPL/SMPL-X registration first, licences stack | VERIFIED CC BY-NC-SA 4.0 (own terms) + VERIFIED SMPL (confirmed by peer review, SUP-20260906-03) |
| SMPL / SMPL-X / `smplx` package | **Do not adopt** for the primary task — `skeleton.py`'s FK already covers the representation this project uses; only revisit if a future track needs mesh-level output | VERIFIED non-commercial, registration-gated, no redistribution (`LANDSCAPE.md` §4.1/4.2/4.3) |
| ControlNet + OpenPose SDXL checkpoint (positioning demo only, not the research track) | **Use via `diffusers`, pretrained, no fork** | Not independently re-verified in this pass — check the specific HF checkpoint's licence card before using in `POSITIONING.md`'s demo; flagged as a to-do, not yet done |

---

## 6. The ablation ladder

Each rung is a pre-registered hypothesis with the metric that decides it, per
`docs/EXPERIMENT_LOG.md`'s template. Numbered to align with future `E-series` entries.

| rung | hypothesis | success criterion | what it settles |
|---|---|---|---|
| **E0 (gate, not a result)** | The vendored Guo et al. evaluator reproduces a published FID to within a stated tolerance on a released checkpoint (e.g. MDM's own) | FID within +/-5% of the paper's reported 0.544 (or the checkpoint actually used) | D-03: is the harness trustworthy at all |
| **E1 — reproduce the original's failure, correctly instrumented** | The original's exact task framing (frame-0-only static pose, corrected decode) produces measurably worse text-alignment than full-sequence generation, on the *same* corrected pipeline | **FID** measurably worse for frame-0-only vs. full-sequence, holding architecture fixed (decisive); R-Precision-top3 reported alongside as a sanity check only, not decisive — see note below | Converts F3 from "moderate effect, ~1.4x dispersion" into an actual measured performance delta — the single most valuable number this project can produce (per `AUTONOMOUS_RUN_PROMPT.md` Stage 3 item 4) |
| **E2 — redundant-vector baseline** | A from-scratch diffusion model, predicting the full 263-d vector (matching MDM/MotionDiffuse's actual representation), reaches FID in the neighborhood of MDM's 0.544 on a laptop-MPS budget | FID within a stated multiple of MDM's number (exact tolerance to be set once E0 establishes measurement noise) | First honest baseline number this project has ever had |
| **E3 — rotation+FK ablation** | Predicting rotations-only + root params, decoded via `skeleton.py`'s FK, beats E2 on FID and/or a direct bone-length-error metric | FID or bone-length-error improves over E2 by more than the seed-to-seed spread (`LANDMINES.md` §7) | Tests D-11's structural-correctness argument empirically, not just logically |
| **E4 — text-encoder ablation** | Token-level CLIP (baseline) vs. DistilBERT (ablation), both measured on a laterality-specific metric in addition to aggregate FID/R-Precision | Laterality metric improves measurably for at least one alternative over pooled-CLIP-equivalent | Tests D-13 empirically; also finally re-measures the original's UNVERIFIED 0.03 cosine-distance laterality claim (`LANDMINES.md` §6) under a real metric |
| **E5 (stretch, not gating) — PoseScript comparison** | If pursued (§2), the same pipeline's static-pose mode, evaluated on PoseScript's own metrics, is compared against E1's frame-0-HumanML3D result | Both numbers reported side by side, explicitly not treated as directly comparable (different eval protocols) | Whether a dataset actually built for static pose changes the picture at all |

**Why FID is the decisive metric and R-Precision is not, across this whole ladder (peer review,
`reviews/REVIEW_QUEUE.md` SUP-20260906-02):** in `LANDSCAPE.md`'s own published-numbers table,
StableMoFusion's R-Prec-top3 (0.841) and MoMask's (0.807) both **exceed** the paper's own
ground-truth "Real" row (0.797). Generated data outscoring real data means R-Precision has no
dynamic range left at the frontier of published results — it can confirm a rebuild is in the
right regime at all, but it cannot be trusted to rank or gate anything close to competitive. FID
still shows an intact, informative range (MDM 0.544 -> MoMask 0.045) and is the metric E0-E3
above actually gate on. This applies to every rung in this table, not just E1.

Every rung's actual results go to `artifacts/<NN>_<name>_record.json` and
`docs/EXPERIMENT_LOG.md`, per `CLAUDE.md`. **No rung is a result until E0 has passed.**

---

## 7. Compute estimate

**What runs on this Mac (Apple Silicon, MPS, `LANDMINES.md` §7 non-determinism applies to all
of these):**
- E0 (evaluator reproduction) — CPU/MPS inference only, no training. Minutes, not hours.
- E1-E4 — small denoiser (the original was ~2.1M params; a redundant-vector or rotation-space
  UNet/transformer at similar scale is realistic on MPS). Rough order of magnitude: hours per
  run on a subset of HumanML3D's ~23k train sequences, not the full published training budgets
  (MDM/MotionDiffuse train for days on datacenter GPUs — matching their exact training length is
  not the goal; landing in a reasonable, honestly-labeled range is).
- **This estimate is a placeholder or judgment**, not independently benchmarked — no model has
  been trained in this repository yet (`docs/00_START_HERE.md` §5). Treat these as first-pass
  planning numbers, to be replaced with actual measured wall-clock times starting at E0.

**What would need a rented GPU:** matching a full published training budget for direct
apples-to-apples comparison against, e.g., MoMask's exact reported number (rather than "in the
neighborhood of," per E2's stated tolerance). Not required for E0-E4 to produce honest,
labeled-as-such results; would matter if a later stage wants to actually contend for a
leaderboard position rather than produce a measured, honestly-scoped baseline.

**Cost:** not estimated here — genuinely unknown until E0-E1 give real wall-clock numbers on this
hardware. Flagging as UNVERIFIED/not yet estimated rather than inventing a plausible-sounding
dollar figure.

---

## 8. Environment — WRITTEN, NOT CREATED

Per `docs/DECISIONS.md` D-09/D-10: this file is written and the create command is printed below.
**No environment has been created.** Ask Joel before running the create command.

```yaml
# environment.yml — T2P-motion-gen-redo, macOS ARM (Apple Silicon)
name: t2p-redo
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch>=2.2          # MPS backend; verify against the version already in mjs_mlcvdl_unified_m5 (2.13.0) before pinning tighter
  - numpy
  - scipy
  - pandas
  - matplotlib
  - pip
  - pip:
      - datasets           # HF datasets, for streaming HumanML3D
      - huggingface_hub
      - transformers        # CLIP / DistilBERT / T5 tokenizers+encoders
      - diffusers           # noise schedulers; ControlNet pipeline for POSITIONING.md's demo track
      - einops
      - pytest
      - tensorboard         # or a lighter local-only tracker; no cloud experiment-tracking service assumed
```

**Create command (DO NOT RUN without asking Joel first, per D-10):**

```bash
mamba env create -f environment.yml
```

**Notes:**
- `mjs_mlcvdl_unified_m5` (already on this machine, used for all of Stage 1's empirical work)
  already has numpy 1.26.4, torch 2.13.0, datasets 4.8.5, huggingface_hub 1.24.0, matplotlib
  3.11.1 — meaning **E0 and probably E1 could start in that existing env with zero new installs**,
  deferring the actual `t2p-redo` env creation until `transformers`/`diffusers`/`einops` are
  actually needed (Stage 3, once the denoiser architecture is being written).
- This env file has not been tested — "written, not created" per D-09, so there is no
  installation to have tested yet. Flagging honestly rather than implying it's been checked.

---

## 9. Risk register

| risk | track affected | kill criterion |
|---|---|---|
| The Guo et al. evaluator doesn't reproduce any published number within a defensible tolerance (E0 fails) | Everything downstream of D-03 | Per D-03's own stated fallback: downgrade every subsequent number to internally-comparable-only, state this loudly in `RESULTS.md` and `LANDMINES.md`, and do not claim comparability to the published ladder in §6 |
| MPS non-determinism (`LANDMINES.md` §7) makes seed-to-seed spread comparable to or larger than the effects being measured (E1, E3, E4) | Any claimed ablation result | If seed spread exceeds the claimed effect size, the comparison is declared not a comparison, per `LANDMINES.md` §7 — no exceptions for a result Joel would like to be true |
| PoseScript's poses turn out to be SMPL parameters (likely) and the licence stack (CC BY-NC-SA 4.0 + SMPL non-commercial/no-redistribution) makes the secondary track legally unusable for anything beyond personal research | E5, and any positioning claim resting on PoseScript | Drop E5 entirely; this does not affect E0-E4, which do not depend on PoseScript |
| The redundant-vector baseline (E2) cannot reach FID anywhere near MDM's 0.544 on a laptop-MPS training budget in reasonable time | The "first honest baseline" deliverable | Report the actual number reached, however far from 0.544, as the honest result — this project's entire premise is that an honestly-measured bad number is worth infinitely more than an unmeasured claim of success. Do not quietly lower the bar without saying so. |
| Time budget for Stage 2/3 research and engineering turns out much larger than planned, given this is a redo not a repair (D-17) — architecture, dataset, and task framing all being reconsidered from scratch is real scope, not a formality | Whole project timeline | No hard kill criterion here — this is Joel's call on how much time to spend, not an engineering gate. Flagging as a genuine open resourcing question, not pretending there's a clean automatic answer. |

---

## OPEN_QUESTIONS carried into Stage 3

1. ~~Whether PoseScript's poses are SMPL parameters~~ RESOLVED — confirmed SMPL+H G format,
   peer review SUP-20260906-03 (§1, §2).
2. The exact tolerance for "within a stated multiple of MDM's FID" (E2) can't be set until E0
   establishes what measurement noise even looks like on this evaluator/hardware combination.
3. ControlNet+OpenPose SDXL checkpoint licence (§5 table, last row) — not yet checked, needed
   before `POSITIONING.md`'s demo track uses it.
4. The two new architectural-level bugs the peer review's supervisor audit found in the original
   project (`docs/LANDMINES.md` §11 — CFG folded into the training loss, making the objective
   trivially satisfiable by ignoring conditioning entirely; §12 — per-batch normalisation of the
   diffusion target plus a mis-indexed timestep applied to the whole batch) were found after this
   spec's drafting began. They don't change any recommendation above, but they strengthen §0
   point 2 and D-02/D-03 generally: a smoothly-falling loss was mathematically guaranteed to fall
   regardless of whether the model learned anything, which is exactly the failure mode an
   evaluation-first harness exists to catch. Worth a forward reference from `EXPERIMENT_LOG.md`'s
   E1 entry once it's run.
