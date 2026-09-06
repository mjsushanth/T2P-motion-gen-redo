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

