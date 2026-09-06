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

### D-11 — Rotation-space output with differentiable forward kinematics · PENDING, strongly favoured
Predicting joint positions forces anatomical validity to be a *loss term* that fights the data
term and still admits invalid poses. Predicting joint rotations over a fixed skeleton makes
bone lengths exactly correct by construction (`LANDMINES.md` §8). This is what the established
text-to-motion models do.
**Rejected (provisionally):** position output plus bone-length loss — the predecessor's design.
**Decided by:** Stage 2's `REBUILD_SPEC.md`, argued explicitly rather than assumed.

### D-12 — Dataset choice is open between corrected HumanML3D and PoseScript · PENDING
HumanML3D is a *motion* benchmark being used for *static poses*, which creates the caption/frame
mismatch in `LANDMINES.md` §3. PoseScript is built for static poses and has published
benchmarks. Current lean: **PoseScript as the primary, corrected HumanML3D retained as the
control** so the size of the decode bug is itself measurable.
**Decided by:** Stage 2, after licence, size and code availability are VERIFIED — not assumed.

### D-13 — Text encoder choice is open · PENDING
CLIP ViT-B/32's pooled embedding is a known-weak signal for laterality and spatial relations
(`LANDMINES.md` §6). Alternatives: token-level CLIP conditioning, a T5-family encoder, or
mirror augmentation. Whichever is chosen, a laterality-specific metric is mandatory so the
failure mode stays visible.
**Decided by:** Stage 2, with evidence fetched rather than recalled.

### D-14 — The three original phases are preserved as reproducible ablation configs · JUDGEMENT
Phase 1 must remain runnable. Reproducing the original failure *with correct instrumentation*
is part of the deliverable: it turns "we had a bug" into a measured delta.
**Rejected:** deleting the broken configuration as dead code.
**Would reverse if:** forensics refutes F1, in which case there is no delta to measure and the
phases become historical context only.

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
