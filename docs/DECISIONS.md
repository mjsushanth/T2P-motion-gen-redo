# DECISIONS — why this project is shaped this way

> **Purpose.** Every non-obvious choice, with the reason, what was rejected, and **what
> evidence would reverse it**. Written so an incoming agent does not "improve" something that
> was already considered and rejected for a stated reason.
>
> Format: **D-nn — decision** · *status* · reasoning · rejected · what would reverse it.
>
> **FORCED** = a fact or a constraint removed the alternatives.
> **JUDGEMENT** = a genuine call that could reasonably have gone the other way. These are the
> ones worth revisiting.
> **PENDING** = deliberately not yet decided; the deciding evidence is named.
>
> Append new decisions; do not renumber existing ones.

---

## Part A — Process

### D-01 — Forensics before rebuild · FORCED
The original project's central defect is a suspected data-decode bug (`LANDMINES.md` §1). If
it holds, most of the original's architectural work was solving problems the bug created, and
porting that work forward would carry the compensations without the cause. If it is refuted,
the whole diagnosis changes. Nothing downstream is worth designing until Stage 1 reports.
**Rejected:** starting from the existing code and fixing things as found — that is how you
inherit an unexamined premise.
**Would reverse if:** never within this project.

### D-02 — Evaluation harness before any model · FORCED
The predecessor produced three months of work and zero measurements. The only structural
defence is to make measurement the first deliverable rather than the last.
**Rejected:** train first and add metrics when there is something to measure — this is exactly
what failed, and it fails because by the time you add metrics you are invested in the answer.
**Would reverse if:** never.

### D-03 — The harness must reproduce a published number before it is trusted · JUDGEMENT
A metric implementation that has never agreed with an external reference is an unvalidated
instrument (`LANDMINES.md` §10). Gate: reproduce one published HumanML3D figure to a stated
tolerance, or explicitly downgrade every number to internally-comparable-only.
**Rejected:** unit tests alone — they prove self-consistency, not correctness of protocol.
**Would reverse if:** no published number turns out to be reproducible from available code and
weights, in which case the downgrade path is taken and stated loudly, not silently.

**Status update (2026-09-06): UNRESOLVED, not passed.** E0a (evaluator sanity, real-vs-real data)
passed and was correctly labelled "NOT the D-03 gate itself." E0b (the actual reproduction
attempt, MDM's released checkpoint against its own published FID 0.544±.044) is the gate this
decision names, and it missed its pre-registered ±5% tolerance (measured FID 1.0731) at a
reduced sample size (n=128, 1 replication — from this project's own measured generation rate,
~39 min/128 samples: one full-scale replication, n≈1,000, would cost ~5 CPU-hours on this
hardware, and the full 20-replication protocol ~100 CPU-hours; the checkpoint's own bundled log
reports ~12 Hrs for that protocol on the *authors'* hardware, a different machine — corrected
here per SUP-20260906-73 after the same figure was found quietly reused as if it were this
project's own cost). Per this decision's own fallback clause: **every number from E1 onward is
internally-comparable-only until D-03 resolves**, stated loudly wherever such numbers appear
(`RESULTS.md`, `docs/EXPERIMENT_LOG.md`). Full diagnostic detail in `docs/EXPERIMENT_LOG.md`'s
E0b entry — the miss is not cleanly attributable to sample size alone (a cheap runtime check
ruled out the most likely driver-bug explanation, mismatched guidance/step count), so this is
recorded as open, not resolved in either direction.
**An earlier statement that this gate was "satisfied by E0a" was wrong** (peer review
SUP-20260906-15, P0) — corrected here and in `docs/EXPERIMENT_LOG.md`/`LEDGER.md` rather than
silently edited out.

### D-04 — The original project directory is read-only · FORCED by instruction
`<ARCHIVE>/` is the archive under study and also Joel's coursework record.
**Rejected:** refactoring in place.
**Would reverse if:** never.

### D-05 — One ledger, named `LEDGER.md` · JUDGEMENT
The Stage-1 worker created `LEDGER.md` before this scaffold existed. Adopting it rather than
introducing `PROGRESS_LEDGER.md` avoids a split record, which is worse than a suboptimal name.
**Rejected:** renaming mid-run (breaks a file another session is actively appending to).

### D-06 — Reviewer and collaborator territory is read-only to the producer · FORCED by policy
`reviews/` and `guidance/` are written by other agents. A reviewer that edits the work destroys
the record that a defect ever existed. The producer replies in `docs/REVIEW_RESPONSES.md` by
finding ID, including declines with reasons.
**Would reverse if:** never. This is a standing rule in Joel's global config.

---

## Part B — Environment and tooling

### D-07 — The original conda environment is not reproduced · FORCED
`dl_t2p_proj_0321.yml` is 12 KB of `win-64` build-pinned strings with CUDA 11.8. This machine
is Apple Silicon. The file is kept as a **provenance record of what the original ran on**, not
as something to install.
**Rejected:** attempting a cross-platform port of the pinned env.
**Would reverse if:** a CUDA machine enters the picture and exact-original reproduction becomes
the goal.

### D-08 — Torch on MPS, never CUDA-assuming code · FORCED
Apple Silicon, no NVIDIA GPU. Consequence: MPS is non-deterministic (`LANDMINES.md` §7), so
every comparison needs multiple seeds.
**Would reverse if:** a rented GPU becomes the primary compute; even then the code must stay
device-agnostic.

### D-09 — Reuse an existing conda env for forensics; defer the project env to Stage 2 · JUDGEMENT
Stage 1 needs numpy/torch/datasets to check a hypothesis, not a curated environment. An env
already on the machine satisfies that with no install and no approval round-trip. The real
project environment is specified in `REBUILD_SPEC.md` and **written but not created** until
Joel approves.
**Rejected:** creating a project env before knowing what the project is.

### D-10 — No installs without asking · FORCED by instruction
Joel's standing rule. Write the env file, print the exact command, stop.

---

## Part C — Technical direction

### D-11 — Redundant-vector baseline, rotation+FK as an ablation · JUDGEMENT (decided 2026-09-06)
**Argued in full in `REBUILD_SPEC.md` §3.** Predicting joint positions forces anatomical
validity to be a *loss term* that fights the data term and still admits invalid poses; predicting
rotations over a fixed skeleton makes bone lengths correct by construction (`LANDMINES.md` §8's
underlying logic, corrected 2026-09-06). **Correction to this decision's original premise:** §8
previously claimed rotation+FK "is what MDM and MotionDiffuse do" — checked directly against both
papers (this session, then independently re-confirmed by the director session against MDM's own
paper, which states "since foot contact and joint locations are explicitly represented in
HumanML3D, we don't apply geometric losses in this section") — **that claim was false.** Both
predict the same redundant vector HumanML3D itself encodes (positions + rotations together), not
rotation-only. **Decision, argued from first principles instead of false precedent:** the
redundant vector is the **baseline** (matches the field, matches what D-03's published-number
target was computed on); rotation+FK (via already-vendored, MIT-licensed
`primary_source/skeleton.py`) is an **ablation rung (E3)**, testing whether making bone-length
correctness structural — a real, F1-grounded hypothesis (bone lengths are a *verified dataset
constant*) — measurably beats the baseline.
**Rejected:** position output plus bone-length loss (the predecessor's design, dominated either
way). Rotation-only as the *baseline* (deviates from the reproducible-published-number strategy).
**Would reverse if:** E3 beats the E2 baseline decisively — promotes to primary, ladder labeling
inverts. This is what the ladder is designed to surface, not a failure of the plan.

### D-12 — Full-sequence text-to-motion on corrected HumanML3D, not static pose · JUDGEMENT (decided 2026-09-06)
**Argued in full in `REBUILD_SPEC.md` §0-§2.** Superseded by a broader reframing than originally
scoped — see D-18. Within that reframing: **corrected HumanML3D, full sequences via
`recover_from_ric`**, decided over PoseScript, because the Guo et al. evaluator
(`EricGuo5513/text-to-motion`, VERIFIED MIT) gives a concrete, fetchable D-03 gate target that
PoseScript's thinner, paper-specific tooling doesn't, and because full-sequence generation
eliminates F3's caption/frame mismatch by construction rather than needing a mitigation.
**PoseScript's poses are confirmed SMPL+H G format** (review SUP-20260906-03) — both
HumanML3D-derived and PoseScript-derived data inherit the identical AMASS/SMPL non-commercial
licence chain (`REBUILD_SPEC.md` §1, `POSITIONING.md` §1), so licence terms do not distinguish
between them; the evaluator-maturity argument does.
**Rejected:** PoseScript as primary (weaker D-03 candidate). Frame-0-only static pose (the
original's framing) — retained as ablation rung E1, not discarded: it converts F3 from a
dispersion ratio into a measured performance delta.
**Would reverse if:** the Guo evaluator turns out unreproducible for a reason not yet known
(E0 fails) — PoseScript's own native metrics become the more pragmatic fallback target.

### D-13 — Text encoder: CLIP token-level baseline, DistilBERT ablation, T5 explicitly rejected · JUDGEMENT (decided 2026-09-06)
**Argued in full in `REBUILD_SPEC.md` §4.** CLIP ViT-B/32's pooled embedding is a known-weak
signal for laterality/spatial relations — evidenced by real external CLIP literature
(arXiv:2311.11477, arXiv:2305.14897), **not** by the original project's own claim, which is now
doubly unverifiable: F6 (CFG folded into the training loss, no conditioning dropout) means the
original's model was never actually required to use the text at all, so its own left-right
failure is not evidence CLIP specifically was the cause. **Decision:** CLIP token-level
(per-token) conditioning as baseline — cheapest fix targeting the specific documented failure
(the pooled bottleneck); DistilBERT as ablation, following TEMOS's HumanML3D-adjacent precedent;
**T5 explicitly rejected** — a HumanML3D-scale model that tried it saw no gain, domain-specific
counter-evidence to Imagen's general T5-over-CLIP finding. Laterality-specific metric (E4) is
mandatory regardless of choice.
**Rejected:** unchanged pooled CLIP (the failure mode this fixes). Full T5 swap (argued against
above). Sentence-transformer embeddings (no source found evaluating spatial/lateral behaviour).
**Would reverse if:** E4's laterality metric shows token-level CLIP still fails badly — DistilBERT
promotes to primary, or explicit mirror augmentation gets added regardless of encoder choice.

### D-14 — Reproduce the original as ONE reference run; drop the phase-by-phase ablation ladder · JUDGEMENT (revised 2026-09-06, supersedes the original text below)
**Original text (2026-09-05, superseded, kept for the record):** "Phase 1 must remain runnable.
Reproducing the original failure with correct instrumentation is part of the deliverable: it
turns 'we had a bug' into a measured delta. Rejected: deleting the broken configuration as dead
code. Would reverse if forensics refutes F1, in which case there is no delta to measure and the
phases become historical context only."

**Why this changed:** the director session's supervisor audit found F6 (CFG folded into the
training loss — the objective has a zero-loss solution requiring no text dependence at all) and
F7 (one timestep steps the whole batch, corrupting the anatomy-loss input for ~95/96 samples per
step). Phase 2's "anatomical breakthrough" enforced a *dataset constant* (F1's corollary) using a
*mis-stepped estimate* (F7); Phase 3's "text conditioning" was trained under an objective that
rewarded ignoring text entirely (F6). **The three-phase narrative does not describe what actually
happened — it describes three independent bugs compounding.** Faithfully reproducing all three
phases would measure the cost of that compounding, not a clean, interpretable delta.
**Revised decision (argued in full in `REBUILD_SPEC.md` §0a):** keep exactly **one**
original-configuration run as a documented historical reference point (provenance, not an
ablation rung with a success criterion). Drop the phase-by-phase ladder as Stage 3/4's organizing
structure in favor of the corrected-pipeline ablation ladder (E0-E5, `REBUILD_SPEC.md` §6), which
isolates the actually useful comparison (E1: task framing, correctly instrumented, free of the
CFG/timestep/normalisation bugs entirely).
**Rejected:** faithfully reproducing all three original phases as ablation rungs (the original
plan) — see reasoning above. Deleting the original configuration as dead code (still kept, as one
reference run).
**Would reverse if:** a future stage finds a way to cleanly separate the three phases' bugs from
each other well enough that a faithful three-phase reproduction would isolate one variable at a
time — not expected, but not ruled out.

### D-18 — Task reframed: full text-to-motion sequences, not static single-pose · JUDGEMENT (new, 2026-09-06)
**Argued in full in `REBUILD_SPEC.md` §0.** The original task (text -> one static pose, frame 0
of a motion sequence) is not inherited as this project's task. **Decision:** the primary research
task becomes text -> full motion sequence, on corrected HumanML3D. This eliminates F3's
caption/frame mismatch by construction (a sequence's caption describes the whole sequence, no
frame-selection problem exists), gives access to a mature, fetchable, MIT-licensed evaluator
(`EricGuo5513/text-to-motion`) with a clear published-number ladder (MDM FID 0.544 -> MoMask FID
0.045) to target for D-03, and lets F1's fix transfer directly and cleanly (same representation,
same decode). F3's finding is not discarded — it becomes ablation rung E1, a controlled
measurement of exactly how much of a performance gap is attributable to task framing versus
everything else, which converts a 1.4x dispersion ratio into an actual measured delta nobody else
can produce, because nobody else made this exact mistake to measure against.
**Rejected:** continuing the original's static-single-pose framing "corrected" — F1 (the original
architecture's design was compensation for a decode bug) and F6 (the architecture was never
meaningfully evaluated at all) together mean there is little reason to inherit the original task
alongside fixing its bugs; per D-17, this is a redo, not a repair.
**Would reverse if:** the director session's review of `REBUILD_SPEC.md` disagrees with this
reframing — the stated fallback is D-12 as originally scoped (PoseScript vs. corrected
HumanML3D, both static-pose); nothing in the rest of the spec is irreversible either way.
**Gate status:** ACCEPTED by the director session's Stage 2 gate review, 2026-09-06 — "the
argument holds... go and flip D-11/D-12/D-13 out of PENDING... and add a D-18."

---

## Part D — Deliverable shape

### D-15 — Notebooks are narrative, `src/t2p/` is machinery · FORCED by policy
The predecessor put its entire system in four notebook cells of 44k, 46k, 52k and 101k
characters, with three near-duplicate re-implementations of the same classes coexisting in one
file. That is the anti-pattern this rule exists to prevent.
**Rejected:** notebook-first development.

### D-16 — The Excel knowledge tier is built last · JUDGEMENT (Joel's instruction)
A colour-zoned workbook is a *presentation* of a finished record. Building it early
manufactures the appearance of results before there are any.
**Would reverse if:** Joel asks for it earlier.

### D-17 — Full redo, not a repair; measurement is method, not mission · JUDGEMENT (Joel's steer, 2026-09-05)
Joel rejected "eval-first" as the project's framing when naming the public repo, in these words:
*"the whole agenda is a proper redo, full redo, it might involve full business idea changes,
research changes, architecture changes."*

The distinction is real and it changes Stage 2's job. D-02 and D-03 (harness before models,
validated against a published number) stand unchanged **as engineering discipline** — without
them a redo cannot be shown to have worked. But they do not define the project. The architecture,
the dataset, the task framing and the business case are all open to **replacement rather than
repair**, and Stage 2 is expected to propose things the original never considered.

Consequence: the positioning work is promoted from an appendix to `POSITIONING.md`, a first-class
Stage 2 deliverable.
**Rejected:** framing the project as "the original, done correctly." F1 makes that framing wrong
anyway — the original's design was largely compensation for a decode bug, so there is little to
do correctly.
**Would reverse if:** Joel asks for a narrow, faithful reproduction instead.

### D-19 — Standing technical authorisation; supersedes D-10 · FORCED (Joel's instruction, 2026-09-06)
Joel, returning to find the build pass paused at a dataset-download approval gate: *"i do not want this in the
future, it has approval to download, experiment, edit - recreate conda mamba new environments,
edit code, replace any gpu - cuda based code, rewrite new code, open jupyter notebooks etc.
do as much it needs."*

**Pre-authorised, no asking:** dataset and checkpoint downloads; creating, recreating and deleting
conda/mamba environments; installing packages; editing and rewriting code; replacing CUDA-assuming
code with MPS; opening and executing notebooks.

**Supersedes D-10** ("no installs without asking") and the ask-first half of D-09. D-09's
substantive point — do not build a project environment before knowing what the project is — stands
as engineering judgement, not as a permission gate.

**Unchanged, because Joel did not address them and they are not technical-work gates:**
`<ARCHIVE>/` stays read-only (it is his graded coursework archive, and nothing in the
rebuild needs to modify it — trivially reversible if he says otherwise); nothing deleted outside
this project folder; no purchases; nothing sent externally; no posting.

**Would reverse if:** Joel says so. Note the cost this gate actually imposed — a full supervisor
cycle elapsed with the build pass idle at a download prompt, which is exactly the failure the autonomous-run
design exists to prevent.

### D-20 — The goal is educational and demonstrable, not commercial · FORCED (author's instruction, 2026-09-06)
The author is a post-graduate student self-funding this work. There is no company, no funding, and
no revenue intent. Stated plainly: *"I WON'T GET any business profit from this... I just need
EDUCATIONAL value, self value, interview value, deep AI research value. ZERO business."*

**Two consequences, and the second is the one that changes the plan.**

**1. The licence wall is moot.** `POSITIONING.md` correctly established that SMPL/AMASS terms bar
training networks for *commercial* use, and treated that as a constraint to route around. With no
commercial intent, the constraint does not bind: SMPL's licence **explicitly permits**
non-commercial research, education, and personal projects. HumanML3D, PoseScript, `smplx` and the
SMPL body models are all fully available for this project's actual purpose. **Do not choose a
representation to preserve a commercial option that is not wanted.** D-11's redundant-vector
decision stands on its own merits; the 2D-keypoint route is now one option among several rather
than the only licence-clear exit.

**2. "No business case" is not an acceptable stopping point.** The author's framing:
*"people want to see a usable PRODUCT out of AI projects... I need you to help me move past
research AFTER FINISHING research architectures, and do an outcome, productive show."* The
research track produces numbers; numbers are not a demonstrable outcome to a non-specialist.
A **local, runnable artifact** is required — explicitly not a deployed web service.
*"Something local is fine too."*

This adds a stage: **Stage 5 — Demonstrator**, after the research ladder completes.
**Rejected:** ending the project at `RESULTS.md`. A metrics table is the *evidence*, not the
deliverable.
**Would reverse if:** never, absent the author saying so.

### D-21 — Sole authorship; no institutional or third-party identifiers · FORCED (author's instruction, 2026-09-06)
This repository is the author's work and is credited solely to them. The original project's
partner was absent throughout and agreed to full credit transfer, which is why that repository has
no forks or pull requests.

Removed from all tracked files 2026-09-06: course code, institution name, group number, any
reference to a partner or co-author, and tool/assistant attributions. The absolute path to the
read-only original-project archive now lives in `.archive_path`, which is gitignored; tracked
documents refer to it as `<ARCHIVE>/`.

**Standing rule:** no institutional identifiers, no third-party names, no assistant attributions
in tracked files. Agents describing their own work use role words (build pass, review pass), not
product names.

### D-19a — The one exception to D-19: system-breaking scale · JUDGEMENT (author, relayed via the build pass 2026-09-06)
Refines D-19 rather than replacing it. The author, restating the authorisation directly to the
build session: *"DONT ASK me those things... unless its some MASSIVE 300GB download or 200GB
environment, something system breaking."*

**Bar:** ask only when an operation risks the machine itself — hundreds of GB of disk, an
environment that would consume the drive, anything that could destabilise the system. **Everything
below that bar proceeds without asking**, including multi-GB datasets and model checkpoints.

Recorded as relayed rather than heard first-hand. It is consistent with D-19 (granted directly)
and strictly *adds* a guardrail above the bar rather than widening the authorisation below it, so
it is safe to act on. If the author contradicts it, his direct statement governs.

### D-22 — Stop chasing E0b. Take D-03's documented fallback and move to E1. · JUDGEMENT (review pass, 2026-09-06)
E0b did not reproduce MDM's published FID. Everything cheap has been checked and the config is
faithful — CFG wrapped once (not doubled), `batch_size` correctly forced to 32 over the
checkpoint's 64, guidance 2.5, 1000 unrespaced timesteps, `p_sample_loop`, `use_ema` default
matching. The build pass verified the first three empirically at runtime; the review pass verified
the rest by reading the driver against MDM's own `eval_humanml.py`.

**What is established:** the evaluator is correct. Ground-truth R-Precision reproduces at 0.7969
against a 20-replication reference of 0.7977 — 0.06 sigma. **That is the property internal
comparisons actually depend on.**

**What is not, corrected 2026-09-06 (review SUP-20260906-25/26/27 — this paragraph originally
treated the FID gap the same way as the R-Precision gap; they turned out to need splitting):**
round 2's rerun accidentally held generation fixed (deterministic under MDM's default seed) while
only the ground-truth reference redrew — and FID moved 1.0731 -> 1.3997, a +30% swing from the
reference redraw alone. **FID at this generated-sample-count (n=128) is underpowered and supports
no conclusion, in either direction** — not confirmed-real, not explained-by-noise, genuinely
undetermined without a larger generated sample. **R-Precision is different:** the same
fixed-generation design showed ground-truth batching noise of only +-2/128 (~0.016), while the
generated excess (0.147) is ~9x that — this anomaly **is** real and substantially exceeds
observed noise, and remains unexplained after every cheap lead. Subset composition is refuted for
the R-Precision anomaly specifically (same distractor pool, same batching — an easier subset
would inflate ground truth too, and it does not, at 0.06 sigma off reference).

**The decision.** Closing this would cost roughly 5 CPU-hours at n~1000, and it buys *comparability
to the published ladder* — not correctness of our own measurements, which the ground-truth
reproduction already establishes. Under D-20 this project's value is educational, interview, and
research, plus a demonstrable artifact. **E1 — measuring what the original project's task framing
actually cost — is this project's own contribution and nobody else's. It is worth more than
matching a number someone else already published.**

So: take D-03's fallback explicitly. **Every downstream number is labelled
internally-comparable-only** until and unless E0b is closed. The generation anomaly is recorded as
a live open question with its evidence, not quietly dropped.
**Rejected:** a second n=128 replication (refuted hypothesis, buys nothing); the ~5-hour n~1000
sweep (right answer to a question that is not this project's question).
**Would reverse if:** a cheap explanation surfaces, or the project later needs to claim
comparability to published numbers — at which point the n~1000 run becomes worth its cost.

### D-23 — E1 as originally specified was a tautology; replaced with an A/B/C conditioning-mismatch design · CORRECTION (review pass, 2026-09-06)
D-22 (and `REBUILD_SPEC.md` §6 before this entry) specified E1 as: compare the original's
frame-0-only static-pose output against full-sequence generation, on the same corrected pipeline,
deciding on FID. **This was wrong, and it was the review pass's own error to catch, not the build
pass's.** A frame-0-only model produces a single static pose; a full-sequence model produces a
sequence. Scoring both through a sequence-only evaluator (`third_party/text-to-motion`'s FID/R-Precision,
built for temporal HumanML3D motion) guarantees the static-pose arm scores badly for evaluator-shape
reasons alone, independent of whether F3's actual conditioning defect (caption truncation, frame
selection) is present. The comparison was rigged to "succeed" before any data existed — a
tautology, not a measurement.

**What F3 actually claims, re-examined:** the original project's defect was a *conditioning
mismatch* — the text caption paired with a single frame does not describe that frame alone,
because (a) captions were truncated to a first-action clause and (b) the frame selected was not
representative of the described action. Both are properties of the *input pairing*, not the
*output space*. Measuring them does not require changing what the model predicts; it requires
holding the output space fixed (full sequence throughout) and varying only the caption/target
pairing.

**The corrected design**, now in `REBUILD_SPEC.md` §6:

| arm | caption | target | isolates |
|---|---|---|---|
| **E1A (control)** | full caption | full sequence | the corrected pipeline, no defect present |
| **E1B** | first-action-clause only (the original's actual truncation rule) | full sequence | caption truncation |
| **E1C (stretch)** | full caption | frame-0-representative conditioning | frame selection |

All three arms share one output space and one evaluator, so the FID/R-Precision comparison is
apples-to-apples — the only variable is the conditioning/pairing under test. **A vs B is the
minimum viable E1**; C runs only if the feasibility measurement below shows budget for it.

**Attribution, per the append-only convention:** the original E1 row's rationale is left in place
in `REBUILD_SPEC.md`'s edit history (git blame) rather than deleted outright; this entry is the
correction record. The error was the review pass's own — the design was proposed and written into
the spec by that session, not by the build pass executing it. Recorded plainly because the
multi-agent review protocol only works if both sides' mistakes go on the record the same way.

**Would reverse if:** a reason emerges that output-space mismatch is actually what F3 is claiming
(re-reading `FORENSICS.md`'s F3 finding does not support this — F3 is about caption/frame pairing,
not architecture) — no such reason has surfaced.

### D-24 — Training is CPU-only. MPS is unusable for this codebase. · FORCED (measured 2026-09-06)
*(Renumbered from a collided D-23 — two sessions appended a new decision at the same number
within the same hour; this is the later-appearing entry in file order, bumped to keep IDs
unique. No content changed. The E1-tautology correction above keeps the D-23 slot.)*

D-08 assumed torch-on-MPS would be the compute target. **Measured and refuted.** MDM's
`diffusion.gaussian_diffusion._extract_into_tensor` indexes a float64 numpy array
(`sqrt_alphas_cumprod`) and moves it to the timesteps' device; MPS refuses float64
(`TypeError: Cannot convert a MPS Tensor to float64 dtype`). Reproduced deliberately on
`--device mps`, then run on `--device cpu`, both reported — the failure was not silently worked
around. This retroactively explains why all E0 work ran CPU-only.

**Measured CPU cost:** 2.252 s/training-step (median of 8 timed steps after warmup, <7% spread),
MDM `trans_enc` defaults, 17.88M trainable params excluding CLIP, batch 32.

**Consequences.** Every wall-clock projection on this hardware is a CPU projection. D-08's
"MPS not CUDA" guidance still holds for *inference and evaluation*, which do run on MPS, but not
for training this architecture. `LANDMINES.md` §7's MPS non-determinism caveat does **not** apply
to CPU training runs — which is a small mercy, since it means seed variation there is genuine
initialisation/ordering variance rather than device noise.

**Would reverse if:** MDM's diffusion schedule is patched to float32 throughout. That is a real
option and it is not large — but it is a change to vendored numerical code, so it needs its own
before/after equivalence check against CPU results before any number produced under it is trusted.

### D-25 — SUP-02's FID-decisive rule is regime-dependent; E1/E2 gate on R-Precision instead · CORRECTION (review pass SUP-20260906-32, 2026-09-06)
SUP-02 (`reviews/REVIEW_QUEUE.md`) established FID as the decisive metric project-wide, because
R-Precision is saturated at the published frontier (StableMoFusion 0.841, MoMask 0.807, both
exceeding the paper's own "Real" ground-truth row of 0.797). That argument is correct **about the
frontier** and was wrongly generalized to every rung in `REBUILD_SPEC.md` §6, including E1/E2.

**The collision:** SUP-31 (this same review pass) established, from three FID values computed on
one bit-identical set of 128 generated motions (1.0731 / 1.3997 / 3.2909, `LANDMINES.md` §14),
that FID's covariance estimate is rank-deficient and unusable as a decision criterion at
generated sample counts this hardware can afford (n~128; n well above the 512-dim embedding
would be needed, and the feasibility projection in `REBUILD_SPEC.md` §7 shows that is not
affordable for a multi-arm, multi-seed matrix). Gating E1/E2 on FID under SUP-02's original,
unscoped rule means those rungs cannot conclude, ever, at this project's compute scale.

**The fix:** SUP-02's rule was written about published-frontier models and does not transfer
unchanged to this project's own early, laptop-scale rungs. **For E1 and E2 specifically,
R-Precision-top3 is decisive** (it estimates no covariance, so small n does not wreck it — proven
directly: ground-truth R-Precision reproduced to three decimals against a 20-replication
reference, batching noise floor ~2/128 ~ 0.016), **FID is reported as secondary**, with its n=128
instability stated inline wherever it appears. E3 onward must re-assess, per rung, which regime
applies (`REBUILD_SPEC.md` §6's regime note) — if a later rung's quality or affordable sample
count approaches the published frontier, SUP-02's original FID-decisive rule re-applies.

**Attribution:** this is the review pass's own correction to its own earlier finding (SUP-02),
caught by the same session before any run used the collided rule, not a build-pass catch. Recorded
per the append-only convention alongside D-22/D-23's precedent for reviewer self-corrections.

**Would reverse if:** a later rung's generated sample count can affordably reach the range FID
needs (roughly n in the high hundreds to low thousands per arm/seed, per the rank-deficiency
argument), at which point FID becomes trustworthy again for that rung specifically and the
regime note in `REBUILD_SPEC.md` §6 should be updated to say so.

### D-26 — E1 stops at E1B; the generation-side comparison is affordably unresolvable, and that boundary is the result · JUDGEMENT (review pass, 2026-09-06)
E1B's raw R-Precision-top3 (0.3438) came in higher than E1A's (0.2969) — opposite the
pre-registered direction. Re-derived independently rather than accepted: the counts behind these
percentages are 44/128 and 38/128, giving a combined standard error of 0.0583 against a gap of
0.0469 — **0.80 sigma, not a signal**. Resolving this at 3 sigma needs roughly 1,780 generated
samples per arm per seed (~9 CPU-hours of generation alone, per arm, per seed) — not affordable
on this hardware, and no amount of additional training-seed measurement changes that, since
binomial sampling noise on the generated count alone already exceeds the observed gap before seed
variance is even added.

**Decision: stop the ladder here.** E1C (frame-selection, designed in `REBUILD_SPEC.md` §6a but
never built), a second seed of E1A or E1B, and any further seeds are not run. All are
**legitimate, pre-registered-as-possible outcomes (SUP-20260906-33 named exactly this branch
before any of E1 ran) that this project cannot afford to resolve, not experiments that failed or
were abandoned** — documented as such in `docs/EXPERIMENT_LOG.md`'s E1B entry rather than left
implicit or silently dropped.

**What this does and does not cost the project.** E1's caption-truncation question is already
answered, model-free, at the retrieval level by the E1-pilot (corpus-wide 0.145-0.157, conditional
~0.27, both 9-17x their own measured noise floors) — a real, well-supported, if narrower-than-
originally-scoped finding. What remains genuinely open is whether that retrieval-space effect
*propagates into generation* — E1B was the only rung testing this, and it is the one question this
hardware cannot afford to answer at adequate power. That is a real gap in what this project can
claim, stated as one, not papered over.

**Rejected:** running E1A/E1B additional seeds anyway "to see" (cannot move the binomial floor,
would only add training-seed variance on top of an already-unresolvable gap — wasted compute, not
a hedge). Building E1C now regardless of E1A/E1B's outcome (the design in §6a is sound but its own
cost is comparable to E1B's, and the same affordability ceiling applies).

**Would reverse if:** a rented GPU or substantially larger compute budget enters the picture,
making ~1,780 samples/arm/seed of CPU generation affordable — at which point E1B could be re-run
at adequate power, and E1C's design (§6a) becomes worth building.


### D-27 — D-24 is REVERSED: MPS runs, at 9.95x CPU. D-26 is reopened pending a generation measurement. · CORRECTION (measured 2026-09-06T21:41Z, author challenge)
The author challenged "not resolvable on this hardware" and asked why MPS was ruled out and why
no Metal/MLX alternative was pursued. The challenge is upheld. D-24's own reversal condition has
now been executed and it reverses.

**D-24's error was not the reproduction — it was the inference.** The float64 failure is real and
C1 reproduced it honestly. What nobody did was test whether the failure was *removable*. D-24
even names the fix in its reversal clause. It sat untested for eight hours while every downstream
cost estimate silently inherited "CPU-only" as if it were a property of the machine.

**The fix.** One line in `diffusion/gaussian_diffusion.py::_extract_into_tensor` — cast before the
device transfer instead of after (`.float().to(device)[t]` rather than `.to(device)[t].float()`).
This is **bit-identical, provably**: indexing is a pure gather, so cast-then-gather equals
gather-then-cast, and the original already discarded the float64 on the next operation. The
float64 schedule *construction* at `gaussian_diffusion.py:165` is untouched, so the "use float64
for accuracy" rationale is preserved. Verified `th.equal` True / max abs diff 0.0 both CPU-vs-CPU
and MPS-vs-CPU.

**Measured — same machine, env, seed, session:** CPU 2.284 s/step (reproduces D-24's 2.252),
MPS 0.231 s/step. **9.95x.** Apple M5 Pro, torch 2.13.0, batch 32, 17.88M params. Re-timed at 40
steps with explicit `torch.mps.synchronize()`: 0.2295 s/step, timer-vs-wall gap 0.00s, no drift.
Run under `PYTORCH_ENABLE_MPS_FALLBACK=0`, so nothing silently fell back to CPU. Step-for-step
losses track CPU to ~1e-3 at the same seed.

**On MLX / Metal alternatives.** They were never investigated, which was also a gap — but they
turn out not to be needed and are now explicitly rejected for this project. PyTorch MPS *is* the
Metal path: it compiles to the same Metal Performance Shaders. Porting MDM to MLX would mean
rewriting a vendored reference implementation whose fidelity to the published model is the whole
basis of D-03's harness gate, trading a one-line provably-equivalent patch for a from-scratch
reimplementation that would need re-validating against every E0 number. Wrong trade. Revisit only
if a specific op is shown to be pathologically slow under MPS.

**Status of D-26 (the E1 stopping rule): REOPENED, not reversed.** D-26's affordability argument
is dominated by *generation* cost (~9.5 min/batch on CPU), and only *training* has been re-timed.
Generation is the same model in a 1000-step loop so a comparable speedup is plausible, but at
batch 32 it may be launch-overhead-bound rather than compute-bound. **The ~54 min/arm/seed figure
implied by naive 10x scaling is not a measurement and must not be quoted as one.** D-26 stands
until generation is timed on MPS and its arithmetic re-derived from measured rates.

**Standing consequence.** `LANDMINES.md` §7's MPS non-determinism caveat comes back into scope for
any run moved to MPS, and D-24's small mercy (CPU seed variance being genuine) lapses with it.
Before any MPS-produced number is trusted, the E0a evaluator sanity check is re-run on MPS and
must land inside the reference band already established on CPU — bit-identity of one helper does
not license bit-identity of a whole training run.

**Would reverse if:** the E0a re-check on MPS lands outside the CPU reference band, in which case
MPS is usable for exploration but every reported number goes back to CPU.

### D-28 — Generation speedup is 5.47x, not 9.95x; D-26 is REVERSED as a bounded null, not annotated; E1 is CLOSED at n=128, no further runs · CORRECTION + JUDGEMENT (measured 2026-09-06T22:1x-23:xxZ, author-prompted)

**Gate result (D-27's own precondition).** `e0_evaluator_sanity_check` re-run on MPS (seed 0):
R-Precision-top3 0.7201923076923077 (CPU) vs identical to all displayed digits on MPS (diff
0.0), matching score diff 4.0e-08, FID diff 7.4e-09 — against a CPU seed0-vs-seed1 spread of
2.7e-03. The seed effect is ~67,000x the device effect. **MPS numbers from this evaluator are
trusted.**

**Generation timing (D-27 step 3, the load-bearing measurement).** Measured directly, not
scaled from training: CPU 551.75s/batch (17.242 s/sample, batch 32, matches the historical
~9.5 min/batch this project's cost estimates were built from), MPS 100.81s/batch (3.150
s/sample). **5.47x** — well short of training's 9.89-9.95x, confirming the predicted
launch-overhead effect at batch 32. **Training and generation speedups are reported
separately from here on; there is no single "MPS speedup" number for this project.** (Caveat:
the CPU arm was timed at n=32/one batch and MPS at n=128/four batches, so CPU warmup is less
amortized; true steady-state CPU is somewhat below 551.75s/batch. Also, `torch.get_num_threads()`
is 6 of this machine's 18 cores — both ratios are "against CPU as this project has actually run
it," not against a thread-tuned baseline.)

**A second real MPS bug, found by running the full pipeline rather than trusting the evaluator
gate alone.** An end-to-end MPS validation (train+generate+evaluate arm A, seed 10, n=128 — same
recipe as the existing CPU record) crashed: `data_loaders/humanml/networks/evaluator_wrapper.py`'s
`get_co_embeddings`/`get_motion_embeddings` (used by every E1A/E1B run, via `EvaluatorMDMWrapper`)
does `.to(self.device).float()` — the identical cast-after-transfer defect D-27 already fixed in
`gaussian_diffusion.py`, in a different file. **The E0a gate above did not catch this because it
exercises a different evaluator class** (`EvaluatorModelWrapper`, `third_party/text-to-motion/`)
whose call site happens to already receive float32 tensors on that specific data path — the same
`.to(device).float()` anti-pattern is present there too (confirmed by grep) but was never
triggered by the E0a check, only by luck of dtype, not by the check having covered it. **Passing
one evaluator's MPS gate does not license trusting a different evaluator class's MPS behavior.**
Fixed with the same provably-equivalent cast-before-transfer reorder in both vendored files
(`motion-diffusion-model` and `text-to-motion` copies of `evaluator_wrapper.py`); verified by
direct unit test (`th.equal` true, MPS-vs-CPU) before re-running the crashed validation. The same
pattern also exists, unfixed, in both vendored `trainers.py` files — not on this project's
execution path (the evaluator networks are loaded pretrained, never retrained here), left as a
known, flagged, unexercised instance rather than patched speculatively.

**Affordability, re-derived from the two measured rates (SUP-20260906-79, independently
recomputed and confirmed correct):**

| n/arm | CPU | MPS |
| --: | --: | --: |
| 128 (what was run) | 2.52h | 0.30h |
| 1,780 (D-26's original target — resolves the *observed noise blip*, 0.0469) | 10.43h | 1.75h |
| full D-26 target, 2 arms x 2 seeds @ n=1,780 | 41.7h | 7.0h |

At the measured MPS rate, even D-26's original (circular) target is one overnight run, not an
impossibility. **D-26's stated reason to stop — "not affordable" — no longer holds as a reason.**

**But SUP-20260906-77 found the deeper defect first, and it does not depend on hardware at
all.** D-26 powered itself to resolve **0.0469** — its own noise reading, at 0.80σ, in the wrong
direction, reproducible from seed alone (E1A seed 10 vs seed 20 differ by exactly 0.0469, the
entire "effect," from seed variance alone). **Powering an experiment to resolve its own noise
blip is circular.** The actual hypothesis-motivated effect size was already measured, by the
E1-pilot, at 0.145-0.157 (retrieval-space truncation cost). Re-derived independently (matches
SUP-77/79 to within rounding): n/arm at 3σ for delta=0.157 -> 159, 0.145 -> 187, 0.117 -> 287,
0.100 -> 392, 0.0469 -> 1,781. **n=128 (what was run) is 70-80% of the n needed for the actual
hypothesis, not 7% of it, as D-26's original framing implied.**

**E1's result, restated (superseding D-26's "affordably unresolvable" framing):** the n=128 run
already establishes a minimum detectable effect of **0.175 at 3σ, 0.117 at 2σ**. **Caption
truncation's retrieval-space cost does not propagate to generation R-Precision at full strength
in this regime: effects >= 0.175 are excluded at 3σ, >= 0.117 at 2σ; whether a smaller effect
(0.05-0.10) exists is open**, and would need n ~ 400-1,600/arm to resolve — a number now
affordable on MPS, but not run, per the decision below.

**Three caveats that travel with this reframing — a bounded null, not a clean one:**
1. Both arms are severely undertrained (3,000 of MDM's 475,000 steps, 0.63%). Both score well
   above chance (0.30-0.34 vs 0.09375), so this is not a floor artefact, but propagation could
   plausibly require a stronger generator to manifest at all — the null is regime-scoped.
2. Retrieval-space cost and generation-space cost are different quantities; attenuation is
   expected on theory, so "not at full strength" is weaker than "absent."
3. Single seed per arm for the primary A-vs-B comparison (plus two supplementary A seeds, seed
   20 and seed 10-on-MPS, the latter a bug-finding byproduct, not a design choice).

**A mechanistic reading, offered as interpretation, not asserted as established — and corrected
before being written down anywhere, per review SUP-20260906-79.** This project's own generator
scores R-Precision-top3 0.30-0.34. The relevant comparison is **MDM's own published, converged,
GENERATED score, 0.611±.007** (`LANDSCAPE.md` line 47) — not the ground-truth/Real row (0.797),
which was the wrong comparison in an earlier draft of this reasoning and is corrected here before
it propagated anywhere. Even MDM's real, converged model is well below ground truth on this
metric (a known field-wide saturation/ceiling effect, `LANDSCAPE.md` §1.3). This project's model,
at 0.63% of that budget, is coarser still. R-Precision at that quality is plausibly driven by
gross features (locomotion, speed, seated) that a first-action-clause truncation *preserves* —
if so, the null is the expected result at this regime, not a measurement failure, and says
something real about where in the pipeline truncation damage does and does not show up. **What
would test it:** repeat at a stronger generator, where finer distinctions become resolvable.

**Regime-scoping note for whoever trains further.** D-25 gates E1/E2 on R-Precision specifically
*because* it does not need covariance and survives small n — correct in this project's own
low-quality regime. `LANDSCAPE.md` §1.3 found R-Precision saturated at the published frontier
(StableMoFusion 0.841, MoMask 0.807, both exceeding the paper's own Real row, 0.797) and
recommended FID there instead. **If this project ever trains toward that frontier (D-27 makes a
full-budget run affordable), R-Precision will stop discriminating exactly where it would matter
most, and the gate must revisit FID per D-25's own regime note before trusting any generation-
quality claim from a converged model.**

**Decision: E1 is CLOSED at n=128. No new arms, no new seeds, no E1C, regardless of MPS
affordability.** A first pass proposed running n=384 x 2 arms x 2 seeds (~2-3 MPS-hours) to
tighten the bound to 0.101σ/0.067σ — the power arithmetic for that proposal was independently
verified correct, but the proposal was retracted (review SUP-20260906-79's own retraction) before
being run, on different grounds: **E1 answers a forensics question about a dead project's
specific bug** (does truncation measurably hurt generation), and a tighter bound on that question
is not worth further compute right now, regardless of how cheap the compute has become. Nothing
was launched under the retracted proposal.

**What this actually validates, stated plainly.** A real, working, already-downloaded model sits
on this machine: MDM's own released checkpoint, already scored by this project (E0b v2) at
R-Precision-top3 = 0.7578 generated against 0.8125 ground truth on this project's own 128-sample
subset — a real text-to-motion model, not a toy. The night's MPS work was spent validating
infrastructure (two real bugs found and fixed) and re-deriving statistics for a 3,000-step
toy comparison, while a working model sat unused. That imbalance, not the MPS work itself, is
the actual finding worth acting on next.

**Would reverse if:** a future stage needs a properly-powered propagation estimate specifically
(not merely "would be nice") — at that point, n~400-1,600/arm on MPS is cheap and the design
above (2 seeds/arm, matching this project's own demonstrated seed sensitivity) is ready to run
un-retracted.

### D-29 — No training-comparison experiments for now: the instrument has to be trusted before it is used to compare · JUDGEMENT (author-directed, 2026-09-07)

**The decision.** No new two-arm training studies, no 475k-step reproduction run, no caption-
augmentation ablation, no training-based experiment of any kind, until further notice. This
applies even though D-27/D-28 just made such runs cheap on MPS — affordability was never the
blocking question.

**Why.** Every training-comparison experiment this project could run right now shares E1's exact
shape: a comparison at a training budget nobody has shown is adequate (this project's own model
tops out at 0.63% of MDM's published budget; even a full 475k-step run would be this project's
first-ever attempt at that scale, with no track record of it working here), measured with an
evaluation instrument this project has already found reasons to distrust in two independent ways
this session — R-Precision saturates at the published frontier and stops discriminating exactly
where a converged model would need it to (`LANDSCAPE.md` §1.3, D-25's own regime note), and the
Guo evaluator's text-motion embedding space collapses to ~1% at full-corpus retrieval, resolving
only within its trained 32-candidate protocol (§1.5 of `RESULTS.md`, `LANDMINES.md` §13). Running
a bigger, more expensive version of E1 before either of those instrument questions is settled
would spend real compute re-deriving the same "was the instrument the problem or the model" doubt
that D-28 just spent a whole night resolving for the retrieval-space finding specifically. **The
instrument work comes first, and it needs no training at all** — it can be answered by probing
the frozen CLIP text encoder MDM already conditions on, and by finding a second, independent
evaluator to cross-check the Guo evaluator against, neither of which requires training a single
model.

**New deliverable format, starting now: executed Jupyter notebooks in `notebooks/`.** Every prior
deliverable in this project has been a script + a JSON record + a markdown write-up. That format
is right for a measurement but wrong for a demonstration meant to be read and be convincing on its
own — a notebook that shows its own internals (the actual CLIP embeddings, the actual similarity
matrix, the actual distribution, not a number quoted from a ledger) is a stronger form of the same
append-only, re-verify-before-trusting discipline this project has followed all along, applied to
presentation rather than just to computation. **A notebook that has never been executed, with no
committed outputs, is not a deliverable** — the same standard this project already applies to
every other claim (VERIFIED means observed here, with the output saved, not asserted).

**Would reverse if:** the instrument questions above are resolved (a second evaluator cross-check
lands, and/or the CLIP spatial-language question is answered one way or the other) and a
specific, pre-registered training comparison is designed against a stated, defensible budget —
at which point D-27/D-28's MPS affordability makes it cheap to actually run.
