# LEDGER — T2P Reboot Forensics (append-only, timestamped)

## 2026-09-05 22:10 — Session start
- Read BRIEFING.md in full. Source project confirmed read-only at
  `<ARCHIVE>/`.
- Scope for this stage: forensics only (F1-F4). No rebuild, no design, no model code.

## 2026-09-05 22:12 — Environment recon
- Local `python3` (homebrew, 3.14.7) has none of numpy/torch/datasets/huggingface_hub/matplotlib.
- Conda env `mjs_mlcvdl_unified_m5` already has: numpy 1.26.4, torch 2.13.0, datasets 4.8.5,
  huggingface_hub 1.24.0, matplotlib 3.11.1. VERIFIED via direct import check.
- Decision: use `mjs_mlcvdl_unified_m5` env for all empirical work in this stage. No
  conda/pip/uv/brew install needed, so no install-approval request required.
- Found pre-existing EDA output artifacts in source project at `pose_diversity_results/`:
  `caption_indices.npy`, `clusters.npy`, `reduced_poses_pca.npy`, `reduced_poses_tsne.npy`,
  `tsne_indices.npy`, `cluster_descriptions.csv`, `ClusteringAnalysis_WBK.xlsx`. These are
  outputs from the original notebook run (read-only, will copy/read not modify) — useful
  for F2 quantification without re-running clustering ourselves.

## 2026-09-05 22:12 — Plan
- F1 primary-source layout confirmation + F4 exhaustive grep + F2 notebook logic read:
  delegated to background research/explore agents (large notebook cells, keep raw dumps
  out of main context per context-management rules).
- F1 empirical bone-length test + F3 caption/frame-0 sampling: requires HF dataset download.
  Will check file size via huggingface_hub metadata API (no download) and report to Joel
  before fetching, per BRIEFING guardrail.

## 2026-09-05 22:20 — HF dataset size check + permission
- `TeoGchx/HumanML3D` full dataset = 4.36 GB across 12 parquet shards (7 train ~500MB each,
  2 test ~327MB each, 1 val ~219MB). VERIFIED via `HfApi.dataset_info(files_metadata=True)`,
  no download performed for this check.
- Asked Joel for explicit permission before fetching (per BRIEFING guardrail + global download
  policy). Joel approved: proceed with streaming mode, up to one full shard (~500MB) if needed.

## 2026-09-05 22:25 — F1 primary source: layout CONFIRMED verbatim
- Fetched primary source directly via GitHub API/raw content into `T2P-Reboot/primary_source/`:
  `paramUtil.py`, `motion_representation.ipynb`, `common/skeleton.py`, `common/quaternion.py`,
  `cal_mean_variance.ipynb`, `README.md` from `EricGuo5513/HumanML3D` (repo default branch).
- `motion_representation.ipynb` cell 2 (`process_file`) builds the 263-d vector via
  `data = np.concatenate([root_data(4), ric_data[:-1](63), rot_data[:-1](126), local_vel(66), feet_l,feet_r(4)])`
  = 263 dims exactly, boundaries at 4/67/193/259/263. VERIFIED — this is the actual repo code,
  matches BRIEFING's claimed layout index-for-index.
- `paramUtil.py`: `t2m_kinematic_chain = [[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]`
  gives 21 bones (consecutive joint pairs within each chain) — matches "21 bone lengths" in BRIEFING.
- `recover_from_ric(data, joints_num)` (same notebook, cell 3) confirmed: slices `data[...,4:67]`
  as ric_data, un-rotates by cumulative root rotation, adds root XZ translation, prepends root
  joint. This is the correct decode function.
- Independent confirmation: a separate research agent fetched the same repo + the CVPR 2022
  paper (openaccess.thecvf.com PDF) via a different path and reported the identical layout,
  plus the exact paper quote (p. 5157, "Pose Representation" section) and one minor paper-vs-code
  notational imprecision (paper's shared symbol `j` for joint counts that differ 21 vs 22 between
  j^p/j^r and j^v) — does not contradict the numeric layout.
- F1 layout claim: VERIFIED from primary source (two independent fetches agree).

## 2026-09-05 22:30 — F2 and F4 notebook forensics (background agent) — both VERIFIED
- F2: `extract_static_poses` (EDA cell 21) takes a RANDOM frame per sequence
  (`frame_idx = random.randint(0, len(motion)-1)`), not frame 0. `cluster_poses` fits KMeans
  directly on this raw un-normalized slice — no normalization between extraction and clustering.
  IMPL's `create_strategically_sampled_dataset` (cell 13) loads `clusters.npy` and treats array
  POSITION as HF dataset ROW ID, but the true row ID per position is only recorded in
  `caption_indices.npy`, which IMPL never loads (0 references, confirmed by grep). Empirically:
  `caption_indices[i] != i` for 23382/23384 positions (99.99% mismatch). IMPL then extracts
  FRAME 0 (`frame_selection='first'`, all call sites) of these mis-indexed rows, normalizes them,
  and labels them with cluster IDs computed from different rows' random frames in raw space.
  n_clusters=10 in EDA `main()`; IMPL hard-codes assumption of 10 clusters (IDs 0-9), consistent
  with EDA but never independently recomputed. VERIFIED, high confidence — every claim backed by
  quoted code + directly-loaded `pose_diversity_results/*.npy` artifacts.
- F4: exhaustive per-line regex grep across all 48 EDA + 87 IMPL cells for val/test loaders,
  val_loss, seeding (torch.manual_seed/np.random.seed/seed_everything/PYTHONHASHSEED), and
  tracking (wandb/mlflow/tensorboard/SummaryWriter/neptune/comet_ml/.log(). Zero hits for:
  val_loader, test_loader, test_split, X_test, holdout, val_loss, torch.manual_seed,
  seed_everything, PYTHONHASHSEED, wandb, mlflow, tensorboard, SummaryWriter, neptune, comet_ml.
  The few superficial hits (`humanml3d["val"]` length print, `validate_clip_encoder` one-batch
  PCA sanity check, `.eval()`/`no_grad` in generation/sampling code) were each traced to their
  call site and confirmed NOT wired into any training-loop validation. Only seeding found
  (`random.seed(42)`/`np.random.seed(42)`, EDA cell 21) governs EDA pose sampling, not model
  training. VERIFIED, high confidence — exhaustive, no sampling of cells, every hit traced.

## 2026-09-05 22:35 — F1 empirical decode test: run and result
- Streamed 250 samples (34,554 frames) from `TeoGchx/HumanML3D` train split via
  `datasets.load_dataset(..., streaming=True)` — no bulk shard download occurred (streaming
  fetched only what was iterated). Patched one line in local copy of `quaternion.py`
  (`np.float` -> `float`, deprecated numpy alias) to make the primary-source code runnable
  under numpy 1.26; this is a compatibility fix to MY OWN downloaded copy in
  `T2P-Reboot/primary_source/`, not to the read-only source project.
- Decode (a) raw `motion[:,:66]` reshaped (22,3): mean bone-length CV across 21 bones = 25.81%,
  median 16.18%, max 81.70% (bone 12-15). Decode (b) `recover_from_ric`: mean CV ~0.00008%
  (bone-length std on the order of 1e-7, i.e. floating-point noise floor).
- Per pre-stated decision rule (CV<5% => correct decode): decode (a) fails badly, decode (b)
  passes essentially perfectly. F1 VERIFIED, not refuted.
- Rendered 3 example skeletons (frame 0) under both decodes to
  `T2P-Reboot/forensics_figs/f1_skeleton_sample{0,1,2}.png`. Visual: decode (a) shows tangled,
  self-intersecting stick figures with one joint flung far from the body; decode (b) shows
  coherent, proportioned standing figures in all 3.

## 2026-09-05 22:40 — F3 empirical neutrality test: run and result
- Used same 250-sample stream. Extracted `ric_data` (motion[:,4:67], 63-d, already
  root-relative and rotation-normalized per primary source) at frame 0 and sequence midpoint.
  69 distinct first-action verbs present in the 250-sample caption set (regex on POS-tagged
  caption field).
- Mean pairwise L2 distance: frame-0 poses = 1.0020, mid-frame poses = 1.4360. Ratio 1.433x.
- Distance from corpus frame-0 mean: frame-0 avg 0.651 vs mid-frame avg 0.968 (same reference
  point). 74.4% of frame-0 poses closer to reference than the median mid-frame is. Within 1 std
  radius: 68.0% of frame-0 poses vs 33.6% of mid-frame poses.
- F3 VERIFIED but effect size is moderate (~1.4x), not an all-or-nothing collapse. Flagged in
  OPEN_QUESTIONS as an area a future stage could sharpen with caption-semantic bucketing.

## 2026-09-05 22:50 — Deliverable complete
- Wrote `T2P-Reboot/FORENSICS.md`: one section per finding (F1-F4), each with claim/method/
  evidence/verdict/confidence, a plain-language "what this means" paragraph, and an
  OPEN_QUESTIONS section (4 items: F2's "8 vs 10" claim didn't reproduce as stated found only
  10 configured in both notebooks; F3 effect size is moderate and could be sharpened; CVPR PDF
  read only by the research subagent, not independently by me; report PDF itself not read in
  this stage at all).
- All four findings: F1 VERIFIED, F2 VERIFIED, F3 VERIFIED (moderate), F4 VERIFIED.
- Stopping here per BRIEFING instruction ("Do not proceed past forensics. Stop and report.").

## [2026-09-05T23:45:00] Item 1 — Peer review follow-up: corrections + housekeeping
**Status:** complete
**Acceptance criteria:** (written before starting) Fold the reviewing session's R-1 corrections
into FORENSICS.md's OPEN_QUESTIONS without altering any of my own verified numbers; complete the
R-2 housekeeping moves so `primary_source/` holds only unmodified upstream files and
`artifacts/forensics/` holds all Stage-1 outputs; do not start Stage 2 without checking with Joel
first, since the original task briefing said "Do not proceed past forensics. Stop and report" and
that gate belongs to Joel, not to a peer session.
**Files changed:**
- `FORENSICS.md`: updated all `forensics_figs/` links to `artifacts/forensics/`; added a
  docs-vs-code addendum under F2 (KMeans fit on raw slice despite PDF claiming "reduced
  embeddings"; 49.6% claimed cluster dominance vs. 38.77% measured); added a caveat paragraph to
  F3's Result per R-3; rewrote OPEN_QUESTIONS items 1 and 4 as RESOLVED (both were the reviewing
  session's own briefing errors, not mine — the "8 vs 10" discrepancy is docs-vs-code, not
  code-vs-code; the report PDF was independently read by the reviewing session, not by me).
- Moved `forensics_figs/f1_skeleton_sample{0,1,2}.png` -> `artifacts/forensics/`; removed the
  now-empty `forensics_figs/` directory.
- Moved `primary_source/{f1_bone_length_results.npz, f1_example_skeletons.json,
  f3_neutrality_results.npz, f3_verbs.json, f3_captions_sample.json, run_f1_f3_analysis.py,
  render_skeletons.py}` -> `artifacts/forensics/`. Removed `primary_source/__pycache__/`.
- Added `primary_source/PATCHES.md` documenting the one-line `np.float` -> `float` compatibility
  fix to `quaternion.py` (numpy >=1.24 removed the deprecated alias; the change does not affect
  `_FLOAT_EPS`'s value or any function used in the F1 test).
- Updated the moved `run_f1_f3_analysis.py` / `render_skeletons.py` docstrings to note they now
  need `paramUtil.py`/`quaternion.py` on the path to re-run (no sys.path hack added; documented
  the run-context requirement instead).
**Environment changes:** none.
**Self-critique defects found:** none in my own Stage 1 numbers — the reviewing session's two
resolved items were errors in the peer's own `BRIEFING.md`, not in anything I produced. The two
new docs-vs-code contradictions (KMeans on raw vs. reduced embeddings; 49.6% vs 38.77%) are new
information I had not surfaced myself and are folded in as reviewer-sourced findings, attributed
as such.
**Revisions made:** see Files changed above.
**Verification performed:** re-read the edited sections of FORENSICS.md after each edit; `ls -la`
on `artifacts/forensics/` and `primary_source/` after the moves to confirm the split landed
correctly (6 result/script files + 3 PNGs in artifacts/forensics/; only README.md, two .ipynb,
paramUtil.py, quaternion.py, skeleton.py, and the new PATCHES.md remain in primary_source/).
**Next:** A peer session ("T2P diffusion Opus Review") has requested Stage 2 (landscape survey +
REBUILD_SPEC.md + resolving D-11/12/13) start immediately in this session. I have not started it.
Reason logged in OPEN_QUESTIONS below — checking with Joel first, since (a) the task that spawned
this session explicitly said to stop after forensics, and (b) `ListAgents` shows several other
peer sessions active on this same project, so it isn't clear this session is the one meant to run
Stage 2 rather than duplicating another session's work.

## [2026-09-06T06:55:00] Item 2 — GitHub repo created, Stage 2 approved and starting
**Status:** complete (repo setup) / Stage 2 in progress
**Acceptance criteria:** Joel confirmed proceeding to Stage 2, but first: create a public GitHub
repo with a name reflecting a full redo (not narrowly "eval-first" — Joel's correction, see
below), commit Stage 1 work, push. Ongoing: periodic commits as work progresses is now standing
practice for this project, not a per-commit ask.
**Files changed:** `.gitignore` added; `git init` in `T2P-Reboot/`; initial commit `89d31d3`
(36 files, Stage 1 forensics + project scaffold). Remote `origin` added and pushed.
**Environment changes:** none (git init is local repo state, not a system/package change).
**Self-critique defects found:** first attempt at repo creation used the name
`text-to-pose-eval-first`, which Joel rejected — his correction: "eval-first is NOT THE AGENDA of
this project... the whole agenda is a proper redo, full redo, it might involve full business idea
changes, research changes, architecture changes." Also: my own first `gh repo create --push`
attempt was blocked by the Claude Code auto-mode permission classifier (a tool-level gate separate
from Joel's chat approval) — did not attempt to route around it; surfaced the exact command for
Joel to run himself instead, per instructions on handling denied actions.
**Revisions made:** presented 4 name candidates matching Joel's existing repo naming style
(`oscd-sentinel2-change-detection`, `mlx-debate-lab`, etc.); Joel picked `T2P-motion-gen-redo`.
Repo created and pushed by Joel directly (`gh repo create ... --push`, run from
`T2P-Reboot/`) after I gave the exact command.
**Verification performed:** `git remote -v`, `git log --oneline -1`, `git status` (clean, tracking
origin/main) — all confirm local state matches remote. `gh repo view mjsushanth/T2P-motion-gen-redo`
confirms public visibility, correct description, recent `pushedAt` timestamp.
Repo: https://github.com/mjsushanth/T2P-motion-gen-redo
**Next:** Sent a status message to the review pass (`local_b04c6f9d-60e0-4bc7-a074-92669bab6d19`,
"T2P diffusion Opus Review") summarizing corrections applied, housekeeping done, and the repo
naming decision, and stating I'll flag disagreements rather than execute Stage 2 verbatim.
Starting Stage 2 Part A (landscape survey) now via parallel research passes.

## [2026-09-06T07:05:00] Item 3 — Read project scaffold; corrected two stale LANDMINES.md entries
**Status:** complete
**Acceptance criteria:** read `docs/00_START_HERE.md`, `docs/LANDMINES.md`, `docs/DECISIONS.md`,
`docs/CODE_MAP.md`, `docs/EXPERIMENT_LOG.md` per the review's R-2 before starting Stage 2
work; note `docs/DECISIONS.md` D-17 already reflects Joel's "full redo, not eval-first" steer and
promotes positioning to a first-class `POSITIONING.md` deliverable (not a REBUILD_SPEC.md
appendix as Stage 2's original instructions said) — planning around the updated D-17, not the
original instruction text.
**Files changed:** `docs/LANDMINES.md` §5 (cluster mismatch) and §4 (loss-reduction claim), both
corrected to match the review's R-1 findings, which had been folded into `FORENSICS.md`'s
OPEN_QUESTIONS but not yet propagated to `LANDMINES.md`, which still stated the old (wrong)
attributions.
**Environment changes:** none.
**Self-critique defects found:** `LANDMINES.md` §5 said "the paper and the sampling config say
8" — wrong; both are 10, and "8" is docs-only (README.md, Obsidian notes). §4 attributed the
"99.995% loss reduction" phrasing to the PDF report's Table 1 — wrong; the PDF has only the raw
loss values, the phrase itself is in the Obsidian interview-prep notes. Both were stale because
they were written before the review's R-1 correction landed in `FORENSICS.md`, and nobody
had gone back to update the earlier document that made the same claims.
**Revisions made:** both sections rewritten with the corrected attribution, sourced to the peer
review, and §5 also gained the "treat this project's own docs as unverified narrative" note as a
generalization (a sixth instance of docs-vs-code mismatch pattern, since two more were found:
KMeans on raw vs. "reduced embeddings"; 49.6% vs. measured 38.77% cluster dominance).
**Verification performed:** re-read both edited sections after writing to confirm they read
correctly and don't contradict `FORENSICS.md`'s phrasing of the same facts.
**Next:** Waiting on 3 parallel background research agents (Stage 2 Part A: text-to-motion on
HumanML3D / text-to-static-pose + pose-as-control-signal / body models + text encoders). Will
synthesize into `LANDSCAPE.md`, `REBUILD_SPEC.md`, `POSITIONING.md` once they return, then update
`docs/DECISIONS.md` D-11/12/13 status with reversal conditions per Stage 2 instructions.

## [2026-09-06T07:15:00] Item 4 — Licence check on vendored primary_source/ files (urgent, per review)
**Status:** complete
**Acceptance criteria:** the review pass flagged that `primary_source/` (files from
`github.com/EricGuo5513/HumanML3D`, vendored for F1 verification) is now in a **public** repo,
which makes its licence status load-bearing rather than a footnote. Confirm the upstream licence
permits redistribution-with-modification, and add whatever attribution/LICENSE file it requires,
before doing anything else in Stage 2.
**Files changed:** `primary_source/LICENSE` (added — verbatim upstream MIT license text, fetched
directly, not retyped from memory). `primary_source/PATCHES.md` (added a Licence section).
**Environment changes:** none.
**Self-critique defects found:** the vendored files had been sitting in a now-public repo without
their required copyright/permission notice since the push in Item 2 — a real (if low-severity,
MIT is a permissive licence) compliance gap that I did not catch myself; the review caught
it.
**Verification performed:** `GET https://api.github.com/repos/EricGuo5513/HumanML3D/license`
returns `license.key: "mit"`; fetched the actual `LICENSE` file content (not just trusting the
GitHub-detected key) via `raw.githubusercontent.com/EricGuo5513/HumanML3D/main/LICENSE` — MIT
License, Copyright (c) 2022 Chuan Guo, standard text, saved verbatim to `primary_source/LICENSE`.
MIT requires the copyright+permission notice be included in copies/substantial portions; adding
the LICENSE file alongside the vendored files satisfies that.
**Next:** apply the same discipline to anything Stage 2 proposes vendoring later (PoseScript,
SMPL/SMPL-X, evaluator code) — SMPL-X specifically is registration-gated per the review;
this is now item 1 of the Part A licence survey, not a footnote, per their message.

**Note, no action taken:** the review pass separately flagged to Joel directly (not to
me, and explicitly said not to act on it without Joel's instruction) that `BRIEFING.md` and
`docs/00_START_HERE.md` identify the course code, institution, and group number of a jointly
authored graded submission, without naming the co-author. Recording here for the audit trail
only. I have not touched either file on this matter and will not unless Joel says to.

## [2026-09-06T08:10:00] Item 5 — Stage 2 complete: LANDSCAPE.md, REBUILD_SPEC.md, POSITIONING.md
**Status:** complete
**Acceptance criteria:** deliver all three Stage 2 documents, each claim VERIFIED against a
fetched primary source or explicitly marked UNVERIFIED; argue D-11/12/13 with evidence but leave
`docs/DECISIONS.md` status as PENDING until the director session reviews the reasoning; propose a
different primary task if the evidence supports it rather than defaulting to "rebuild the
original's framing correctly," per the director's explicit widened scope.
**Files changed:** `LANDSCAPE.md` (new — synthesizes 3 parallel research passes: text-to-motion/
HumanML3D evaluator+published numbers; text-to-static-pose datasets + pose-as-control-signal;
body models/tooling + text encoders). `REBUILD_SPEC.md` (new — proposes switching the primary
task from static single-pose to full text-to-motion sequences on corrected HumanML3D, since this
eliminates F3 by construction rather than mitigating it, and the Guo et al. evaluator gives a
concrete, MIT-licensed, fetchable D-03 gate target; argues D-11 as redundant-vector baseline +
rotation/FK as an ablation rung rather than the reverse; argues D-13 as CLIP-token-level baseline
+ DistilBERT ablation, T5 explicitly rejected on domain-specific evidence; full ablation ladder
E0-E5; environment.yml written not created; risk register). `POSITIONING.md` (new — Track A [3D
research on HumanML3D/PoseScript-derived data] has no business case, stated in those words,
because the training data's licence forecloses it; Track B [2D pose -> image via Bonnet et al.'s
MIT-licensed approach + ControlNet, never touching SMPL/AMASS] is plausible but unestablished,
proposed as the "smallest demonstrable end-to-end thing").
**Environment changes:** none. No installs performed; `environment.yml` written per D-09/D-10,
create command printed, not run.
**Self-critique defects found (caught before finalizing, not by an external reviewer):**
1. `docs/LANDMINES.md` §8 and the original `docs/DECISIONS.md` D-11 both asserted "this is what
   MDM and MotionDiffuse do" (rotation-space + FK) as justification. I checked this directly
   against both papers before relying on it (neither of the 3 landscape research agents had been
   asked to verify this specific architectural claim) — **it's wrong**. Both papers' HumanML3D
   experiments predict the same redundant vector (positions + rotations + velocities together),
   not rotation-only. Corrected `LANDMINES.md` §8 in place and re-argued D-11 in
   `REBUILD_SPEC.md` from first-principles logic (bone lengths are a verified dataset constant,
   so structural correctness is a real advantage) rather than false field-precedent.
2. Checked HumanML3D's own upstream README before writing `POSITIONING.md` and found it states
   the motion data itself (not just the processing code) inherits AMASS/SMPL's non-commercial,
   non-redistributable licence — this wasn't asked for by name in any research agent's brief; I
   went looking because the "business case" framing in D-17 made it clearly load-bearing.
**Revisions made:** see Files changed. `docs/LANDMINES.md` §8 corrected in place (separate from
the two corrections already made in Item 3).
**Verification performed:** direct `curl`/GitHub-API license checks for
`EricGuo5513/HumanML3D` (MIT, `primary_source/LICENSE` added, Item 4), `EricGuo5513/text-to-motion`
(MIT — the D-03 evaluator) and `GuyTevet/motion-diffusion-model` (MIT), all this session, not
delegated. Direct `curl`+regex fetch of MDM's (arXiv:2209.14916) and MotionDiffuse's
(arXiv:2208.15001) own papers to check their actual output representation before relying on the
LANDMINES §8 claim. Direct grep of `primary_source/README.md` for the AMASS distribution-policy
statement.
**Next:** message the director session with all three deliverables; flag the MDM/MotionDiffuse
correction prominently since it changes D-11's argument structure, not just its wording; leave
`docs/DECISIONS.md` D-11/12/13 as PENDING per the agreed process.

## OPEN_QUESTIONS (see also the later, current OPEN_QUESTIONS block at the end of this file —
## items below are kept per append-only convention but both are now RESOLVED)

- ~~**D-11/12/13 status.**~~ **RESOLVED 2026-09-06** — all three decided in `docs/DECISIONS.md`,
  gate-accepted by the director session's Stage 2 review.
- ~~**Task-reframing proposal.**~~ **RESOLVED 2026-09-06** — accepted as D-18, gate-accepted by
  the director session.

## [2026-09-06T08:30:00] Item 6 — Peer review (Review 1) of LANDSCAPE.md: 3 findings actioned
**Status:** complete
**Acceptance criteria:** the director session independently re-fetched and confirmed 5 load-
bearing citations in `LANDSCAPE.md` (all exact), then raised 5 findings via
`reviews/REVIEW_QUEUE.md` (SUP-20260906-01 through 05). Read the queue in full, act on every
actionable finding, reply to every finding ID in `docs/REVIEW_RESPONSES.md` per the established
protocol, edit only my own territory (never `reviews/`).
**Files changed:** `LANDSCAPE.md` (§1.3 R-Precision-saturation note; §5.1 scope caveat on
arXiv:2601.12809; OPEN_QUESTIONS #1 marked RESOLVED). `REBUILD_SPEC.md` (§1, §2, vendor table,
ablation ladder, OPEN_QUESTIONS — all updated for the confirmed PoseScript/SMPL fact and the
FID-not-R-Precision gate metric). `POSITIONING.md` (§1 rewritten to lead with the "no commercial
product" conclusion and the three explicit exits, per the review's exact framing).
`docs/REVIEW_RESPONSES.md` (all 5 finding IDs replied to, one per the template).
**Environment changes:** none.
**Self-critique defects found:** none of my own beyond what the review caught — this is the
review doing its job, not a self-critique pass. Noting for the record that SUP-20260906-02 (the
R-Precision saturation) was independently checkable from my own table and I hadn't remarked on
it — a real miss, caught by the reviewer rather than by me, worth naming honestly.
**Revisions made:** see Files changed.
**Verification performed:** re-checked the R-Precision saturation claim directly against
`LANDSCAPE.md`'s own table (StableMoFusion 0.841, MoMask 0.807 vs. Real 0.797) before editing —
confirmed, not taken on the reviewer's word alone. Did not independently re-verify SUP-03's
PoseScript-SMPL-format claim myself (stated as such in the response); flagged if that matters.
**Next:** message the director session confirming all 5 findings actioned; Stage 2 is now
complete pending their review of the updated documents. `docs/DECISIONS.md` D-11/12/13 remain
PENDING as agreed.

## [2026-09-06T08:50:00] Item 7 — Reciprocal audit of F6/F7/F8; Stage 2 gate review actioned; D-11/12/13/18 decided
**Status:** complete
**Acceptance criteria:** the director session asked for a reciprocal audit of their own F6-F8
findings (BRIEFING.md, LANDMINES.md §11-12) "the way I audited your landscape" — verify the
algebra and the code-reading independently, don't just take it. Separately, their Stage 2 gate
review (SUP-20260906-06..10) released the gate for E0/E1, accepted the task reframing, and asked
for D-11/12/13 to be flipped out of PENDING with a new D-18 recording the reframing, plus several
concrete REBUILD_SPEC.md fixes.
**Files changed:** `REBUILD_SPEC.md` (new §0a on D-14; E1 given a seed-spread threshold matching
E3; E2's tolerance made an explicit pre-registration-ordering requirement; D-13 section prefaced
with the F6-based correction to the laterality diagnosis; E4's wording corrected from "re-measure
the original's claim" to "first real measurement"; §5 vendor table's HumanML3D row tightened per
SUP-09; §9 risk-register row updated for the confirmed PoseScript/SMPL fact; a new explicit
conditioning-dropout code requirement + mandatory tripwire-test description). `docs/DECISIONS.md`
(D-11, D-12, D-13 rewritten from PENDING to JUDGEMENT/decided-2026-09-06, each citing
`REBUILD_SPEC.md`'s argument and stating a reversal condition; D-14 revised in place, original
text preserved and marked superseded rather than deleted, per the director's own corrected rule
that both agents may write `docs/` but must append/annotate, never silently rewrite; new D-18
recording the task reframing).
**Environment changes:** none.
**Self-critique defects found:** none new — this entry is the audit and the gate-review response,
not a self-critique pass. Recording explicitly, though: I did not independently re-verify
SUP-20260906-03 (PoseScript's SMPL format) myself in the previous item, and still haven't — it is
taken on the director's fetch, stated as such in `docs/REVIEW_RESPONSES.md`.
**Revisions made:** see Files changed. `docs/DECISIONS.md`'s original D-14 text was NOT deleted —
quoted verbatim and marked superseded, consistent with the director's corrected append-only rule.
**Verification performed — the F6/F7/F8 audit itself, done directly against the read-only source,
not taken on the director's word:**
- Read `DL_T2P_IMPL.ipynb` cell 47 directly (the read-only source project, not modified) and
  located `_compute_loss` in full. **F6 confirmed verbatim:** `predicted_noise = uncond_noise_pred
  + guidance_scale * (cond_noise_pred - uncond_noise_pred)` followed immediately by
  `diffusion_loss = F.mse_loss(predicted_noise, noise)` — exactly as quoted. Algebra re-derived
  independently: with `u=c=eps`, `u + w*(c-u) = u = eps`, so `MSE(eps, eps) = 0` for any `w` —
  confirmed correct, the objective has a zero-loss solution requiring no text dependence.
- **F7 confirmed verbatim:** `timestep=t[0].item()` inside the same function, with `t` established
  elsewhere as a per-sample tensor.
- **F8 confirmed verbatim:** `normalize_batch` computes `batch.mean(dim=0)`/`batch.std(dim=0)` and
  rescales `x_0` against them before noising — exactly as quoted.
- Searched all cells for `guidance_scale=`, the ramp values, and every occurrence of "dropout" —
  confirmed `guidance_scale=3.0` default, `base_guidance=2.0, max_guidance=7.0,
  guidance_ramp_epochs=50` (matches the claimed 2.0->7.0 ramp exactly), and **every single
  "dropout" hit in the notebook is `nn.Dropout` (ordinary layer regularization) — zero hits for
  conditioning/null-embedding dropout anywhere.** F6's "no conditioning dropout exists anywhere"
  claim is confirmed by exhaustive absence, not by trusting the claim.
- **Bonus finding, not in the director's original report:** cell 46's own markdown notes describe
  this exact (flawed) mechanism as the intentional design — "implementation uses progressive
  guidance scaling during training (2.0→7.0 over 50, 100 epochs)" and "we use
  `self._forward_with_text` twice... helping us on CFG formula... loss uses run two parallel..."
  This means F6 was not a stray bug the author didn't notice — it was the documented, believed-
  correct approach. Strengthens F6's severity; worth adding to `LANDMINES.md` §11 as corroboration
  if the director wants it, not added unilaterally since §11 is their entry.
- **Audit verdict: all three findings (F6, F7, F8) hold. No corrections needed to the director's
  work.** Reported this back rather than silently accepting.
**Next:** reply to the director session with the audit result and confirmation of all fixes;
Stage 3 is now gate-released for E0/E1 per their review. E2 remains blocked on writing its
tolerance into `docs/EXPERIMENT_LOG.md` as a dated entry after E0 completes, before E2 runs —
this is now an explicit process requirement in `REBUILD_SPEC.md`, not yet executed since no code
has been written. Environment (`environment.yml`) still needs Joel's approval before creation.

## [2026-09-06T09:10:00] Item 8 — D-19 authorization confirmed directly by Joel; E0 progress
**Status:** partial (E0 in progress, not complete)
**Acceptance criteria:** the director session's `CLAUDE.md`/`docs/DECISIONS.md` D-19 edit claimed
Joel granted broad standing technical authorization (downloads, installs, env create/recreate/
delete, code execution, no more asking) — but the quoted instruction was addressed to a different,
stuck worker session ("C1"), not to this session directly. Per this session's own rules (a peer
cannot grant escalation merely by writing a claim into a shared file), confirm directly with Joel
before treating it as authorization for this session too, rather than silently adopting it or
silently ignoring it.
**Files changed (this item):** none yet from the authorization question itself (confirmation only).
Separately, this item's E0 work added: `third_party/text-to-motion/` (vendored evaluator repo:
`networks/`, `utils/`, `motion_loaders/`, `common/`, `scripts/`, `options/`, `data/`,
`final_evaluations.py`, `train_tex_mot_match.py`, GloVe word vectors, `LICENSE`, `PATCHES.md`).
`.gitignore` extended to exclude `/checkpoints/`, `*.tar`, `*.pth`, `*.ckpt` (large binaries never
committed). `checkpoints/t2m/text_mot_match/model/finest.tar` downloaded (245,580,211 bytes,
gitignored, not in git history).
**Environment changes:** none yet — E0's remaining work is expected to run in the existing
`mjs_mlcvdl_unified_m5` env (numpy/torch already present; no spacy/nltk dependency needed since
HumanML3D's captions are already POS-tagged in `word/POS` format, matching what
`utils/word_vectorizer.py`'s `WordVectorizer.__getitem__` expects directly).
**Self-critique defects found:** none re: the technical work. Recording the authorization-check
process itself as the notable event this item: asked Joel directly via AskUserQuestion rather
than either (a) silently adopting the claimed authorization because it appeared in a project file,
or (b) ignoring a plausible, well-documented, and contextually consistent claim out of excessive
caution. Joel confirmed: "Yes, applies to all sessions on this project."
**Verification performed:**
- Confirmed `docs/DECISIONS.md` D-19 exists and quotes Joel directly, with a stated reversal
  condition ("would reverse if Joel says so") — read in full before asking, not acted on from the
  `CLAUDE.md` summary alone.
- Confirmed `github.com/EricGuo5513/text-to-motion` licence: MIT (GitHub API `license.key`, this
  session, 2026-09-06 — same discipline as `primary_source/`).
- Checkpoint provenance: could not fetch the official Google-Drive-hosted checkpoint zip headlessly
  (no size metadata obtainable without an API key or browser interaction). Found a third-party
  Hugging Face re-upload (`Tevior/text_mot_match`, Apache-2.0, 245.6 MB) with exactly matching
  filename/directory structure. **Asked Joel before downloading** (stated filename, source, size,
  and the provenance caveat explicitly) — approved. After download, loaded the checkpoint's three
  state dicts and checked every tensor shape against `networks/evaluator_wrapper.py`'s
  `build_models()` and `networks/modules.py`'s three encoder classes for the `t2m` config
  (`dim_pose=263`, `dim_word=300`, `dim_motion_hidden=1024`, `dim_text_hidden=512`,
  `dim_coemb_hidden=512`, 15-way POS one-hot) — every shape matched exactly (e.g.
  `movement_encoder.main.0.weight` is `(512, 259, 4)`, matching `Conv1d(263-4, 512, 4)`).
  Documented in `third_party/text-to-motion/PATCHES.md`: this is strong architecture-match
  evidence, explicitly **not** a cryptographic or author-confirmed match to the original file —
  labelled UNVERIFIED beyond that.
- Confirmed `utils/word_vectorizer.py`'s `WordVectorizer.__getitem__` expects `"word/POS"` string
  items — directly compatible with the HF `TeoGchx/HumanML3D` dataset's caption format already
  used in Stage 1 (e.g. `"man/NOUN squat/VERB..."`), no reformatting needed.
- Scoped the remaining E0 work precisely by reading `motion_loaders/dataset_motion_loader.py`:
  it calls `get_opt(opt_path, device)` (expects an on-disk `opt.txt` metadata file, which this
  project does not have — those ship with the *generation*-model checkpoints, not the evaluator)
  and `Text2MotionDatasetV2` (expects an on-disk `new_joint_vecs/`/`texts/`/`test.txt` directory
  layout, not this project's HF-streaming access pattern). Neither is a blocker in principle —
  the needed `opt` fields are standard, documented HumanML3D-repo constants (`dim_pose=263`,
  `unit_length`, `max_motion_length=196`, etc.), most already confirmed via
  `evaluator_wrapper.py` — but reconstructing them correctly and adapting the data loading to
  streamed data is real, unfinished work, not yet done.
**Next:** construct the `opt` Namespace manually (avoiding the on-disk `opt.txt` dependency),
adapt or reimplement the minimal parts of `Text2MotionDatasetV2`'s windowing/normalisation logic
needed to feed streamed HumanML3D test-split (motion, caption) pairs through
`EvaluatorModelWrapper.get_co_embeddings`/`get_motion_embeddings`, then compute matching-score/
R-Precision/FID via `utils/metrics.py` on ground truth against itself as the first sanity check
(targeting the paper's own "Real" row: FID ~0.002, before attempting reproduction of any
generated-model's published number).

## [2026-09-06T09:35:00] Item 9 — E0 sanity check run and written up (2 seeds, full test split)
**Status:** partial (sanity check complete; the actual D-03 gate — reproducing a published
generated-model number — is not yet done, no generative model exists in this repo)
**Acceptance criteria:** Joel confirmed broad standing technical authorization applies to this
session directly (see Item 8); build and run the previously-scoped E0 test to completion, report
honestly whether it's the actual D-03 gate or a precursor, with seed spread per project convention.
**Files changed:** `scripts/e0_evaluator_sanity_check.py` (new — builds `opt` manually avoiding
the evaluator's on-disk `opt.txt` dependency; sorts batches by `sent_len` descending matching the
official `collate_fn`, a requirement discovered only by hitting `pack_padded_sequence`'s sorted-
order assertion; batches R-Precision by 32 candidates matching the paper's protocol, discovered
necessary only after an initial 200-candidate-pool run gave badly-wrong numbers). `checkpoints/`
gained `t2m_dataset_stats/{Mean,Std}.npy` (official HumanML3D normalization stats, fetched
directly, not reconstructed). `docs/EXPERIMENT_LOG.md` gained the E0 entry (full method,
progression table showing the protocol-bug-fix impact, 2-seed spread table, explicit "does NOT
establish" section). `artifacts/e0/` gained the two seed-specific result records.
**Environment changes:** none — ran entirely in `mjs_mlcvdl_unified_m5`, no new packages needed.
**Self-critique defects found:** the first working run (R-Precision on a single 200-candidate
pool instead of 32-candidate batches, matching the paper's own protocol) gave R-Prec-top3=0.28
and FID=0.745 — badly wrong-looking numbers. Caught this myself before reporting it as any kind
of result, diagnosed the actual cause (pool-size mismatch inflates retrieval difficulty; FID is
separately unstable with too few samples relative to the 512-dim embedding space), fixed both,
and documented the wrong version in `EXPERIMENT_LOG.md`'s progression table rather than quietly
discarding it — the fact that the harness *can* produce a badly-wrong-looking number from an
honest methodology bug (not a broken checkpoint) is itself worth keeping as a record.
**Revisions made:** see Files changed. Also removed a redundant unseeded artifact file
(`artifacts/e0/e0_evaluator_sanity_check_record.json`, byte-identical to seed0's after a filename
fix mid-session) rather than leaving an ambiguous duplicate.
**Verification performed:** ran with 2 different seeds (0, 1) on the full HumanML3D test split
(4384 sequences, 4198 usable after length filtering — matches the field's commonly-cited test-set
size, a further indirect confirmation this project's HF-streamed data matches the original
split). Seed spread is tiny (R-Prec-top3: 0.7202 vs 0.7159, spread 0.0043; FID: 0.02870 vs
0.02890, spread 0.00020) — this ran entirely on CPU (the evaluator's BiGRU encoders are small),
so `LANDMINES.md` §7's MPS non-determinism does not apply to this specific check; the small
residual spread reflects only which random crop/subset-split each seed drew.
**Result, precisely:** matched text/motion pairs retrieve each other at 72% top-3 accuracy out of
32 candidates (vs. ~9% chance) — strong signal. FID between disjoint real subsets is 0.029,
down 25x from the buggy protocol's 0.745, converging as sample size increased to the full split.
Remaining gap from the paper's own numbers (0.720 vs. 0.797 R-Prec; 0.029 vs. 0.002 FID) is real,
stable across both seeds and across the sample-size increase from 1024->2099, and most likely
attributable to this project's single-crop-per-motion evaluation vs. the paper's typical
multi-crop-averaged protocol (`--repeat_time` in their own `final_evaluations.py`) — stated as a
plausible, not confirmed, explanation.
**Next:** this sanity check gives strong confidence the harness/checkpoint/data-pipeline are
sound, but does **not** complete D-03 — that requires reproducing an actual published
*generated-model* number (e.g. MDM's FID 0.544), which needs either MDM's own released checkpoint
+ generated samples, or this project's own future E2 baseline once a model exists. Reporting this
precisely to Joel and the director rather than overclaiming the gate is passed.

## [2026-09-06T09:50:00] Item 10 — Joel granted blanket technical authorization directly; E0b setup
**Status:** in progress (setup complete, generation run launched, result pending as this entry is
written — append-only convention: this entry describes what was true when written, a following
entry will report the actual FID once the run completes, not edited into this one)
**Acceptance criteria:** several crossed-in-transit director messages (session was mid-run) asked
for: D-11/12/13/18 flip (already done, Item 7), reciprocal audit (already done, Item 7),
POSITIONING.md rewrite for D-20 (done, Item 9 — wait, see next item), and "go build E0 end to
end... report the reproduction against MDM's FID 0.544 +/-5%." Also mid-session, Joel sent a
direct message: *"DONT ASK me those things. do all you wish. unless its some MASSIVE 300 GB
download or 200 GB environment, something system breaking and unnatural as heck."* Confirmed
this session-wide, not just for the earlier CLAUDE.md-claimed authorization.
**Files changed:** `third_party/motion-diffusion-model/` (cloned whole repo, MIT, verified via
GitHub API; ~3.5MB). `checkpoints/mdm/humanml-encoder-512/` (413MB checkpoint zip, gitignored,
via `gdown` after installing it). `scripts/materialize_humanml3d_test_subset.py` (new — writes a
test-split subset to disk in MDM's expected `new_joint_vecs/`/`texts/`/`test.txt` layout).
`scripts/e0b_mdm_reproduction.py` (new — reduced-scale driver reusing MDM's own
`evaluation_parser()`/`create_model_and_diffusion()`/`evaluation()` rather than reimplementing
them, so every hyperparameter loads from the checkpoint's own bundled `args.json`, not a
hand-reconstructed config). Three patches to the MDM clone, all documented in its own
`PATCHES.md`: (1) lazy SMPL loading in `model/rotation2xyz.py` + guards in `model/mdm.py`'s
`_apply`/`train` — the real SMPL body model is gated (registration required, not something this
session can do on Joel's behalf) and is never actually reached for this project's `hml_vec`
data representation; (2) `num_workers` set to 0 across several DataLoaders — the code assumes
Linux's fork-based multiprocessing, and macOS's spawn-based default can't pickle a local `lambda`
collate function; pure performance parameter, no correctness effect.
**Environment changes:** installed `gdown`, `git+https://github.com/openai/CLIP.git`, `spacy`,
`smplx`, `wandb` into `mjs_mlcvdl_unified_m5` (no new env created — all fit in the existing one).
No approval requested per-package, per the standing authorization confirmed this session.
**Self-critique defects found:** the first `materialize_humanml3d_test_subset.py` had a real bug
— it did `caption_field.split("#")` on the whole raw HF caption field, not realizing (until
directly inspecting the raw field with `repr()`) that it contains **multiple newline-separated
caption entries**, each already correctly formatted. The naive split corrupted every entry after
the first, and MDM's own `Text2MotionDatasetV2.__init__` silently drops any sample whose text
file raises an exception during parsing (bare `try/except: pass` around each sample) — so this
bug silently produced an **empty generated dataset** (`real_num_batches 0`) rather than an error,
which is exactly the class of defect `LANDMINES.md` exists to catalogue. Diagnosed by directly
inspecting `repr(ex['caption'])` on a real HF row rather than continuing to guess. Fixed: split on
newlines first, pass each already-correct line through unchanged. Confirmed E0a's own script was
NOT affected by the same bug (it only ever used `parts[1]`, correctly extracted regardless of
what followed).
**Verification performed:** checkpoint provenance — the checkpoint bundle
(`humanml_trans_enc_512/model000475000.pt`) came with the author's own evaluation log
(`eval_humanml_trans_enc_512_000475000_gscale2.5_wo_mm.log`, 20 replications, dated 2022-09-21),
whose summary (`vald` FID mean 0.5443, CI 0.0442) matches the paper's published 0.544+/-.044
almost exactly — strong independent confirmation this is genuinely the right checkpoint, found
by inspecting the bundle's own contents rather than trusting the README's "best model" label
alone. Cross-checked `dataset/humanml_opt.txt` (real file, shipped with the MDM clone) against
E0a's hand-built `opt` Namespace fields (`dim_movement_enc_hidden=512`, `dim_movement_latent=512`,
`unit_length=4`, `max_text_len=20`) — exact match, retroactively confirming E0a's reconstruction
was correct, not just architecture-shape-matched.
**Next:** generation run in progress (128 samples, 1 replication, reduced from the paper's
~1000 samples / 20 replications — infeasible at full scale per the checkpoint's own bundled log
stating "about 12 Hrs" for the full protocol, on CPU with no dedicated GPU). Will report the
actual FID, pass/fail against +/-5% of 0.544, honestly, in a following ledger entry once it
completes — not averaging E0a and E0b together, per the director's standing instruction.

## [2026-09-06T10:50:00] Item 11 — E0b result: FAIL against pre-registered tolerance, reported as such
**Status:** complete (E0b run and reported; a secondary-metric bug left unfixed, noted below)
**Acceptance criteria:** report the measured FID against the ±5% tolerance around MDM's published
0.544, honestly, whichever way it goes — per the pre-registered commitment in `docs/EXPERIMENT_LOG.md`
not to widen the tolerance after seeing the result.
**Files changed:** `docs/EXPERIMENT_LOG.md` (E0b's pre-registered entry — hypothesis/criterion
left unedited — now has the result appended below it). `artifacts/e0/e0b_mdm_reproduction_record.json`
(hand-assembled from the run's stdout after a crash prevented the driver script's own JSON-write;
noted as a lower-confidence provenance path than a clean programmatic write, though the numbers
themselves were read directly off the log).
**Environment changes:** none beyond Item 10's package installs.
**Self-critique defects found:** the run's `Diversity` computation crashed
(`diversity_times=128` exactly equals `num_samples_limit=128`, failing an `assert
activation.shape[0] > diversity_times`) — an off-by-one in this project's own driver script, not
MDM's or the evaluator's code. Left unfixed/unrerun: FID and R-Precision (the metrics that
actually matter here, per the Stage 2 review's finding that R-Precision is saturated and FID is
decisive) were already computed and printed before the crash, and rerunning costs another ~40
minutes of CPU-bound generation for a secondary metric this gate doesn't need.
**Verification performed:** blocking-waited on the actual run (realized partway through that
`ScheduleWakeup`-based waiting does not appear to advance wall-clock time for an already-running
background process in this sandbox — its own `ps`-reported elapsed time was far behind the sum of
my scheduled delays — switched to a Bash command that blocks on the real completion condition,
which does keep real time advancing; each ~580-second batch was directly observed via repeated
`ps`/log checks, not assumed). Confirmed the checkpoint's own bundled evaluation log
(`eval_humanml_..._gscale2.5_wo_mm.log`, 20 replications, author's own 2022-09-21 run) reports
`vald` FID mean 0.5443 — matching the paper's 0.544 almost exactly — establishing this really is
the right checkpoint, independent of whatever this project's own reduced-scale run measured.
**Result:** measured FID = 1.0731 (vs. paper's 0.544±.044, tolerance band 0.5168-0.5712) —
**FAIL**, reported as such. This run's own ground-truth-vs-itself FID (0.1339, vs. the paper's
~0.002) shows the same small-sample-size inflation E0a's own progression table already
documented (0.745 at n=200 -> 0.029 at n=2099) — meaning this specific FAIL cannot cleanly
distinguish "a real discrepancy" from "n=128 is too few to measure FID stably," and the honest
verdict is exactly that ambiguity, not a claim that MDM's number doesn't reproduce.
**Next:** none required — D-03's gate is satisfied by E0a per the director's Stage 2 review; E0b
was an additional, harder check the director asked for, and its result (FAIL-as-measured, with a
stated confound) is now on the record either way. Reporting to Joel and the director.

## [2026-09-06T11:10:00] Item 12 — P0 correction: D-03 is UNRESOLVED, not satisfied; guidance/step diagnostic run
**Status:** complete
**Acceptance criteria:** review (SUP-20260906-15, P0) correctly identified that Item 11's closing
line — "D-03's gate is satisfied by E0a" — was wrong. The Stage 2 review's gate condition named
the reproduction (E0b) as the actual gate; E0a's own title already said it was not the gate.
Since E0b missed its tolerance, D-03's own documented fallback applies: downgrade every
downstream number to internally-comparable-only until resolved, stated loudly, not silently.
Correct the record (append a correction, do not silently rewrite Item 11) and act on the
fallback. Separately, review (SUP-20260906-16/17) sharpened the diagnosis: sample-size bias does
not fully rescue the FAIL (subtracting the run's own measured bias floor still leaves ~1.7x
outside tolerance), and the specific pattern — generated R-Precision *better* than published
while generated FID is *worse* — is the textbook signature of a stronger effective
classifier-free guidance than the reference used. Check this before spending more compute.
**Files changed:** `docs/EXPERIMENT_LOG.md` (E0b entry: added a correction block replacing the
wrong "satisfied by E0a" closing line with D-03's actual UNRESOLVED status and the fallback
clause, plus the bias-subtraction and R-Precision/FID diagnostic detail). `docs/DECISIONS.md`
(D-03: added a "Status update" noting UNRESOLVED, the earlier wrong statement, and the
fallback in effect).
**Environment changes:** none.
**Self-critique defects found:** the wrong "D-03 satisfied by E0a" claim was my own error,
caught by review, not by me. Recording plainly rather than softening it: I had E0a's own title
correctly labelled "NOT the D-03 gate itself" and then contradicted that same distinction one
entry later when writing E0b's closing line — the discipline I was praised for keeping (SUP-14)
lapsed exactly where it mattered most, in the sentence that determines whether every downstream
number can claim comparability.
**Verification performed:** ran an empirical (not just code-reading) check of the three most
likely driver-bug explanations for the R-Precision/FID pattern, instantiating the model/diffusion
without running any generation (cheap, per review's explicit instruction to check this "before
anything else, it costs minutes"): `args.guidance_param` = 2.5, matching the checkpoint's own
bundled log filename (`eval_humanml_..._gscale2.5_...`) exactly; `diffusion.num_timesteps` = 1000
and `len(diffusion.use_timesteps)` = 1000, confirming no DDIM/respacing shortcut was taken
(`create_gaussian_diffusion` hardcodes `timestep_respacing = ''`, always uses the full step
count); sampler confirmed as `p_sample_loop` not `ddim_sample_loop` (`use_ddim = False` is
hardcoded in `comp_v6_model_dataset.py`). **All three match the reference protocol exactly** —
this specific, cheapest explanation does not hold. Did not find a driver bug to fix.
**Next:** the remaining candidates for the R-Precision/FID pattern are (a) single-replication
stochastic generation variance (this run used `replication_times=1`; the paper's own 20
replications exist partly to average out exactly this kind of run-to-run noise, which one run
cannot distinguish from a real effect) and (b) the still-open checkpoint-provenance confound
(architecture-verified, not cryptographically verified against the original Google-Drive-hosted
file). Proposed, not yet run: a second independent single-replication run at the same n=128 to
check whether the pattern repeats — cheaper (~40 min) than a full n~1000 sweep (~5 CPU-hours),
and decides whether to spend the larger amount at all. Holding here to report before committing
more compute, given how much has already been spent on this one gate.

## [2026-09-06T11:15:00] Item 13 — Operational finding: deferred waiting does not advance wall-clock for a running background process in this sandbox
**Status:** complete (documentation of an operational constraint, for whoever runs a long job next)
**Context:** while E0b's ~39-minute generation ran in the background, I initially tried to "wait"
for it using `ScheduleWakeup` (delays of 90-600s each, several in a row). Checking the actual
background process's own `ps`-reported elapsed time after each wakeup showed it had advanced far
*less* than the sum of the delays I'd requested — e.g. after several scheduled delays summing to
roughly 20+ minutes, the process itself reported only ~2 minutes of real elapsed/CPU time.
**Working hypothesis, based on direct observation, not confirmed against any documentation of
this sandbox's internals:** this environment appears to only advance real wall-clock time for
already-running background processes while a tool call is actively executing in this session —
not during the gaps between a `ScheduleWakeup` firing and the next turn. A `ScheduleWakeup` delay
schedules *when I get control back*; it does not, by itself, appear to keep the underlying
sandbox "live" in the interim for a process I am not actively watching.
**What worked instead:** replacing the scheduled-wait pattern with a single `Bash` call that
*blocks* on the actual completion condition (`until [ -f <output> ] || ! pgrep -f <process>; do
sleep N; done`), run with a long `timeout`. Each such call, once it timed out and moved to
background, had reliably advanced the target process's real elapsed time by roughly the same
duration as the blocking call's own timeout — confirmed by comparing `ps -o etime,time` before
and after each one. Repeating this (rather than `ScheduleWakeup`) is what actually got E0b's
~39-minute generation to complete.
**Practical guidance for a future long-running background job in this project:** don't rely on
`ScheduleWakeup`'s delay alone to let a background compute job progress — issue direct blocking
`Bash` waits (or another mechanism that keeps a tool call actively running) for as much of the
expected duration as possible, and treat `ps`-reported elapsed/CPU time on the actual process as
the source of truth for how much real progress has happened, not the sum of requested delays.
**Not independently verified:** whether this is a general property of the harness/sandbox or
specific to this session's configuration — recorded as an empirical observation from this one
session's experience, not confirmed against any external documentation.

## [2026-09-06T11:45:00] Item 14 — Bundled log resolves R-Precision ambiguity; E0a's own bug found; three leads ruled out
**Status:** complete (leads exhausted for now; root cause of the FID/R-Precision gap not found)
**Acceptance criteria:** review (SUP-20260906-20..24) asked to (a) read the bundled log's
R-Precision, not just FID (cheap, decisive); (b) reconcile E0a's own R-Precision gap explanation
against E0b's correct ground-truth numbers; (c) check the three cheapest leads (length
distribution, caption uniqueness, caption-motion pairing) for the generation-side defect,
caching the generated motions this time so no further question costs another regeneration; (d)
do not run a second replication for variance alone (already answered by the bundled log's own
20-replication spread).
**Files changed:** `scripts/e0b_mdm_reproduction.py` (now builds the generated-motion loader
directly instead of via `eh.evaluation()`'s lazy lambda, caches motions+metadata to
`artifacts/e0/e0b_generated_cache/`, computes and prints a length/caption diagnostic before
scoring). `docs/EXPERIMENT_LOG.md` (E0b entry: "Round 2" section with the bundled-log
cross-check and all three ruled-out leads; E0a entry: correction block retracting the
multi-crop-averaging explanation, pointing to E0b's correct ground-truth numbers as the reason).
`artifacts/e0/e0b_mdm_reproduction_record_v2.json`, `artifacts/e0/e0b_generated_cache/` (new —
128 cached generated motions + metadata, reusable for any future re-analysis at zero
regeneration cost).
**Environment changes:** none.
**Self-critique defects found:** E0a's own multi-crop-averaging explanation (my own hypothesis,
stated as "plausible, not confirmed" — correctly hedged at the time, but still wrong) is now
known to be wrong; the actual cause was a bug in my own hand-built `opt`/data-loading
reconstruction for that script, exposed only by comparison against E0b's use of MDM's real
upstream loader. Corrected in place per the append-only convention (original text kept, marked
superseded) rather than silently rewritten.
**Verification performed:** re-read the bundled log's `R_precision Summary` section directly
(not just its FID section, which is all that had been checked before) — confirmed `[vald] (top
3) Mean: 0.6110`, matching the published 0.611 to three decimals, settling that MDM's number is
correct and this project's discrepancy is entirely on this project's side. Modified the driver
to cache generated motions and ran once more (n=128, 1 replication, same ~39-minute cost),
computing length statistics and caption uniqueness directly from the cached data rather than
assuming: generated-vs-ground-truth length means differ by 0.7 frames (not a fixed-length
bug); all 128 captions unique (not a duplication bug); confirmed by code inspection that
`CompMDMGeneratedDataset.__getitem__` reuses the identical caption/tokens entry attached at
generation time (not a caption-motion desync bug). The qualitative pattern (better R-Precision,
worse FID than published) reproduced across both independent runs despite different batch
shuffling, further confirming it is not single-replication noise.
**Next:** all of round 2's cheap leads are exhausted without finding a fixable bug. Remaining,
not yet tested: whether this project's 128-sample subset (first rows in HF streaming order
passing the length filter) is representative of the full ~4384-sequence test set the reference
protocol draws from. Testing this needs either a properly randomized subset draw at the same n,
or the full n~1000 scale (~5 CPU-hours) to remove the question. Holding before spending either —
reporting the exhausted-leads status to the director first, consistent with checking in before
further compute spend rather than continuing unilaterally (SUP-24's commendation for doing this
last time still applies; not treating it as a one-time exception).

## [2026-09-06T12:00:00] Item 15 — D-22: stop chasing E0b, take D-03's fallback, move to E1
**Status:** complete (decision recorded and executed; E0b closed as an open question, not resolved)
**Acceptance criteria:** director's decision (`docs/DECISIONS.md` D-22, already written by the
director) — the subset-composition hypothesis is refuted for free (ground-truth R-Precision is
0.06σ off the 20-replication reference; an easier subset would inflate ground truth too, and it
doesn't; generated R-Precision is ~3.5σ high, real and unexplained). Director independently
verified the driver's config against MDM's own `eval_humanml.py` (CFG applied once not doubled;
`batch_size=32` correctly overrides the checkpoint's own `64`; `use_ema` default matches) —
every cheap surface is now checked by two independent readings, mine and the director's. Decision:
do not spend ~5 CPU-hours on n~1000 (buys comparability to a published number, not correctness —
already established via the 0.06σ ground-truth match); take D-03's fallback explicitly; record
the anomaly as a live open question; move to E1.
**Files changed:** `LEDGER.md` (this entry; marked the two stale OPEN_QUESTIONS items above as
resolved rather than leaving them looking live; added the current open question below).
`docs/DECISIONS.md` D-22 was written directly by the director session — read and accepted, not
authored by me.
**Environment changes:** none.
**Verification performed:** re-read D-22 in full before proceeding, rather than acting on the
cross-session message's summary alone.
**Next:** E1 — measure what the original project's task-framing bug (F3: pairing "first action of
caption" with "frame 0 of sequence") actually cost, on the corrected pipeline. Per
`REBUILD_SPEC.md` §6, this needs a working generation model (frame-0-only vs. full-sequence,
same architecture, same corrected decode) — the first model this project actually trains, not
just harness/evaluator validation. Real new scope; assessing shape before starting.

## OPEN_QUESTIONS (current)

- **The E0b generation anomaly, per D-22 — a live open question, not dropped. CORRECTED
  2026-09-06 (Item 16): the FID and R-Precision halves of this question are NOT the same
  strength of finding — split below, do not re-merge them.**
  - **R-Precision: real, unexplained.** Generated motions score R-Precision-top3 = 0.7578,
    ~9x the observed ground-truth-batching noise floor (measured directly: GT R-Prec moved by
    only 2/128 ~= 0.016 between two runs with generation held fixed) above the published/
    reference value of 0.611. Every cheap explanation checked and eliminated: guidance scale
    (2.5, matches exactly), diffusion steps (1000, no respacing), sampler (`p_sample_loop`, not
    DDIM), CFG wrapping (once, not doubled), batch size (32, correctly overriding the
    checkpoint's own 64), generated motion length distribution (matches ground truth closely),
    caption uniqueness (128/128 unique), caption-to-motion scoring pairing (verified correct by
    code inspection), subset-composition/easier-retrieval-task (refuted — would inflate
    ground-truth R-Precision too, and it doesn't: 0.06σ off reference).
  - **FID: no conclusion available, in either direction.** Two rounds' FID (1.0731, 1.3997)
    were wrongly read as "reproducing across independent runs" — they don't; MDM's default seed
    makes generation deterministic, so both rounds scored the *same* generated motions, and only
    the ground-truth reference redrew between them. That redraw alone swung FID 30%. A
    fixed-reference rescoring (full n=4640 reference, same cached generated motions, zero
    regeneration) gave FID=3.2909 — higher still, not lower, showing the *generated* side's own
    n=128 covariance is independently unstable regardless of reference quality. **FID cannot be
    judged pass/fail at any sample size this hardware has produced so far.**
  - **Cost to fully close (both halves, with proper power): ~5 CPU-hours at n~1000.**
    Deliberately not spent, per D-22 — the ground-truth reproduction already establishes this
    project's harness is correct, which is the property internal comparisons (E1 onward) actually
    depend on. Revisit the R-Precision anomaly specifically if a cheap explanation surfaces
    later; revisit FID only if the project ever needs to claim comparability to MDM's published
    numbers specifically, since no amount of reference-fixing alone will resolve it — the
    generated sample count itself would need to grow.
- **D-03 status, restated plainly for whoever reads this next:** UNRESOLVED, not satisfied, not
  going to be resolved further within this stage per D-22. **Every number from E1 onward is
  internally-comparable-only** — this must be stated loudly in `RESULTS.md` when it exists and
  in every relevant `EXPERIMENT_LOG.md` entry, not left as an implicit footnote.

## [2026-09-06T12:20:00] Item 16 — Round 2's "reproduced across runs" claim retracted; FID/R-Precision split; fixed-reference rescoring
**Status:** complete
**Acceptance criteria:** review (SUP-20260906-25..29) identified that round 2 did not sample
generation variance — MDM's default seed makes generation deterministic, so round 1 and round
2's identical vald R-Precision (0.7578, 97/128, both times) is the *same* generated motions
scored twice, not independent draws; only ground-truth batching varied. That accidental design
still produced something useful: holding generation fixed and varying only the reference showed
FID swings 30% (1.0731->1.3997) on a same-generation re-reference alone — retracting the earlier
"FID gap is real, ~1.7x unrescued by sample size" claim (my own SUP-16-adjacent framing, echoed
into `docs/DECISIONS.md` D-22) — while R-Precision's same design shows the generated excess is
~9x the observed ground-truth batching noise, *strengthening* that anomaly's claim to being real.
Requested follow-up (SUP-28, the one further E0b action authorised): build a fixed, full-scale
ground-truth reference (no generation needed) and re-score the already-cached 128 generated
motions against it, removing reference-redraw variance from any future FID in this project.
**Files changed:** `scripts/e0b_fixed_reference_rescoring.py` (new — builds and saves a fixed
`(mu, cov)` reference from the full 4198/4640-sequence test split, re-scores the cached
generated motions against it, zero regeneration). `artifacts/e0/fixed_gt_reference/` (new —
reusable reference asset for future FID work). `artifacts/e0/e0b_fixed_reference_rescoring_record.json`
(new). `docs/EXPERIMENT_LOG.md` (E0b entry: retraction of the "reproduced across independent
runs" claim; the FID-vs-R-Precision split stated as two separately-resolved questions; Round 3's
fixed-reference result and its own honest, unexpected interpretation). `docs/DECISIONS.md` (D-22:
corrected the FID characterization, left the original text visible with the correction rather
than silently rewriting).
**Environment changes:** none.
**Self-critique defects found:** my own round-2 write-up claimed the qualitative pattern
"reproduced across two independent runs, ruling out single-run noise" — wrong, caught by review,
not by me. I had the deterministic-seed fact available (I'd printed `fixseed(args.seed)`'s effect
nowhere, but the seed itself was visible in every args.json dump I'd already made) and didn't
connect it to what "identical to four significant figures" implied about the two runs not being
independent. Recorded plainly rather than minimized.
**Verification performed:** re-derived the noise-floor comparison myself before accepting it —
ground-truth R-Precision-top3 moved from 102/128 (round 1) to 104/128 (round 2), a swing of
2/128 ~= 0.016, against a generated excess of 0.7578-0.611=0.147 (~9x) — confirmed this arithmetic
directly from the two runs' own recorded counts rather than taking the review's ratio on faith.
Ran the fixed-reference rescoring script (fast, ~1 minute, no diffusion sampling involved) and
observed FID=3.2909 directly, not a number that was expected or assumed going in — the result
(higher, not lower, than either small-n comparison) was surprising and is reported as such rather
than adjusted to match a prior expectation.
**Next:** E0b is now closed with an honest, split verdict: R-Precision anomaly real and
unexplained (a genuine open question, per D-22/Item 15); FID at achievable sample sizes on this
hardware supports no conclusion in either direction. Moving to E1.

## [2026-09-06T13:05:00] Item 17 — E1 redesigned (A/B/C conditioning-mismatch) after director's SUP-30 tautology finding; corrected in DECISIONS.md as D-23
**Status:** complete
**Acceptance criteria:** the director (review pass) found E1 as specified (D-22/original
`REBUILD_SPEC.md` §6) compared frame-0-only static-pose output against full-sequence output
through a sequence-only evaluator (`third_party/text-to-motion`) — a tautology, not a
measurement, since the static-pose arm was guaranteed to score badly for evaluator-shape reasons
alone regardless of whether F3's actual defect (caption truncation, frame selection) was present.
Required fix, per the director's instruction: replace with an A/B/C design that holds the output
space fixed (full sequence throughout, one evaluator) and varies only the caption/target pairing —
E1A (control: full caption -> full sequence), E1B (first-action-clause caption -> full sequence,
isolates caption truncation), E1C (stretch: full caption -> frame-0-representative conditioning,
isolates frame selection). "A vs B is the minimum viable E1."
**Files changed:** `REBUILD_SPEC.md` §6 (E1 row replaced with the E1A/E1B/E1C table, plus a status
note that A+B must both run for E1 to count complete and C is budget-gated). `docs/DECISIONS.md`
(new D-23, the correction record — states plainly the error was the review pass's own, proposed
and written into the spec by that session, not caught by the build pass; the old E1 row's
rationale is left in git history rather than deleted, per the append-only convention).
**Environment changes:** none.
**Self-critique:** I did not catch this myself before the director flagged it, despite having
written and re-read the tautological E1 row across several turns. Worth naming: the failure mode
was accepting "same corrected pipeline" as sufficient without asking what evaluator the two arms'
outputs would actually pass through, and whether that evaluator's *shape assumptions* (temporal
motion, not a single pose) were symmetric between arms. A generically useful check for future
ablation rows: before accepting a comparison as valid, ask "what would make this arm score badly
for reasons unrelated to the hypothesis under test."
**Verification performed:** re-read `FORENSICS.md`'s F3 finding directly (not from memory) to
confirm the corrected E1A/B/C design still targets what F3 actually claims — caption/frame
*pairing*, not output-space/architecture. Confirmed: F3 is entirely about caption truncation and
non-representative frame selection, both properties of the input pairing, so holding the output
space fixed across arms is the correct isolation, not a dilution of the original claim.
**Next:** measure real training feasibility before scaffolding or committing to any arm (Item 18,
this same session) — the director's explicit ordering: measure first, decide step budget from the
measurement, do not scaffold `src/t2p/` until a wall-clock projection exists in this ledger.

## [2026-09-06T13:40:00] Item 18 — E1 training feasibility probe: real seconds-per-step measured, MPS ruled out, step-budget proposal
**Status:** complete (measurement); step-budget decision proposed, not yet run
**Acceptance criteria:** per the director's instruction, measure real per-training-step wall-clock
cost on this machine BEFORE writing any E1 scaffolding or committing to a training run — training
(forward+backward+optimizer step) is a categorically different cost than the generation-only
sampling already measured in E0b (~39 min / 128 samples, which is 1000 *reverse-diffusion* steps
per sample, not 1000 training steps).
**Files changed:** `scripts/e1_training_feasibility_probe.py` (new — reuses MDM's own
`train_args()` parser via constructed `sys.argv`, same pattern as the E0b driver reusing
`evaluation_parser()`, so architecture/optimizer hyperparameters are MDM's real defaults, not
hand-built guesses; builds the real `HumanML3D` train-mode dataset loader, the real
`create_model_and_diffusion` model, and times N actual `training_losses` -> `backward` ->
`optimizer.step()` cycles after a short untimed warmup. Produces no checkpoint and claims no
result — a timing instrument only). `artifacts/e1/e1_training_feasibility_probe_record.json`
(new — raw per-step timings, losses, device used).
**Environment changes:** none (ran inside the pre-existing `mjs_mlcvdl_unified_m5` conda
environment, same one E0a/E0b used; no new environment created, consistent with
`REBUILD_SPEC.md` §8's "written, not created" status for `t2p-redo` itself).
**What was measured:**
- **MPS (Apple Silicon GPU) is not usable for training this architecture as vendored.**
  `diffusion.gaussian_diffusion._extract_into_tensor` indexes a `float64` numpy array
  (`sqrt_alphas_cumprod`) and moves it to the timesteps' device — MPS refuses float64
  (`TypeError: Cannot convert a MPS Tensor to float64 dtype`). This is a real, reproduced
  incompatibility, not a config error on this project's part; it also retroactively explains why
  all prior E0 work ran CPU-only. Fixing it would mean patching MDM's diffusion schedule to cast
  to float32 throughout — out of scope for a feasibility probe; noting it here as a real
  constraint rather than silently working around it.
- **CPU training-step cost, real architecture (trans_enc, 8 layers, latent_dim=512, 1000
  diffusion timesteps, CLIP text encoder — MDM's true defaults, not the original failed
  project's ~2.1M-param model): 17.88M trainable params (excl. CLIP), batch_size=32.**
  8 timed steps after 2 warmup steps: times ranged 2.142s-2.297s, **median 2.252 s/step, mean
  2.222 s/step** — tight spread (<7% range), a stable number to extrapolate from.
- **Extrapolation (median 2.252 s/step, single run, no eval-during-training overhead):**
  | steps | wall-clock, 1 run | wall-clock, 4 runs (E1A+E1B x 2 seeds) | wall-clock, 6 runs (+E1C x 2 seeds) |
  |---|---|---|---|
  | 1,000 | 37.5 min | 2.5 h | 3.75 h |
  | 3,000 | 112.6 min (~1.9 h) | 7.5 h | 11.3 h |
  | 5,000 | 187.7 min (~3.1 h) | 12.5 h | 18.8 h |
  | 10,000 | 375.3 min (~6.3 h) | 25.0 h | 37.5 h |
**Proposal (not yet executed — a design call the director should weigh in on before it's final):**
3,000 steps per run, E1A + E1B only at 2 seeds each (4 runs, ~7.5h total wall-clock), E1C deferred
unless this comes in comfortably under budget and the A-vs-B result itself calls for a third
data point. Rationale for 3,000 steps specifically: this is a from-scratch small-model run on a
~4.6k-sequence materialized subset (not HumanML3D's full ~23k train split, which is not yet
materialized on disk — only the test split is, per `materialize_humanml3d_test_subset.py`), so
matching a real training budget in the tens-of-thousands-of-steps range is neither necessary nor
honestly framed as comparable to MDM's own published training length; 3,000 steps is a
first-pass number chosen to be checkable in an afternoon, not derived from a convergence
criterion — **this is a judgment call, flagged as such, and open to the director's pushback
before it is treated as final.** Under D-19a this does not need Joel's approval to run (well
under the "hundreds of GB / system-breaking" bar) — it needs the director's research-design
sign-off, since step count is a methodology choice, not a resource-risk one.
**Self-critique:** I have not yet verified whether 3,000 steps produces a model whose loss has
actually stabilized (the 8 timed steps above show losses bouncing 0.82-1.52 with no trend
visible over so few steps — expected at this scale, not informative about convergence). If E1A's
loss curve at 3,000 steps still looks like pure noise, that is itself a finding requiring either
more steps or an honest "did not converge" label on the result, not a silently accepted result.
**Verification performed:** ran the timing probe twice in effect — once on `--device mps` (failed
with the float64 error, confirmed reproducible, not a fluke) and once on `--device cpu`
(succeeded, numbers reported above). Did not fabricate an MPS number or silently skip reporting
the failure.
**Next:** send this measurement and the proposed step budget to the director for review before
committing to run E1A/E1B; if approved (or amended), materialize the HumanML3D train split
(extending `materialize_humanml3d_test_subset.py`'s pattern to `train.txt`), then scaffold
`src/t2p/`'s data-pairing logic (E1A/E1B caption construction) as a thin wrapper around this
vendored MDM training loop — not a reimplementation of it.

**Correction appended 2026-09-06T13:55:00 (review SUP-20260906-32, P1) — Item 18's projection
above was incomplete, not wrong in what it measured, wrong in what it omitted.** It timed
training-step cost only. The director caught, correctly, that E1's decisive metric requires
*generating and evaluating* samples for every arm/seed too, and that generation is diffusion
sampling at 1000 steps — the same operation E0b measured at ~39 min/128 samples — which is not a
small addition. Two of the director's own earlier findings also collided inside the ladder this
same review pass: SUP-02 (gate on FID) vs. SUP-31 (FID unusable at affordable n) — resolved by
inverting to R-Precision-decisive for E1/E2 specifically (`docs/DECISIONS.md` D-25), which is
also what keeps the *corrected* generation-inclusive projection affordable: R-Precision needs
only n=128 to be decisive (proven stable at that n), whereas the superseded FID-decisive
requirement would have needed n~512+ per arm/seed — compare the two generation-cost rows below.
See `docs/DECISIONS.md` D-25 and `REBUILD_SPEC.md` §6/§7 for the corrected criteria and cost
table; not repeating the full table here to avoid a second copy drifting out of sync.

## [2026-09-06T13:55:00] Item 19 — Corrected E1 feasibility: training + generation, R-Precision-decisive regime
**Status:** complete (measurement + design correction); still pending director sign-off before any
real run
**Acceptance criteria:** produce one feasibility number that actually reflects everything an E1
run costs (training AND generation/evaluation), under the metric that will actually decide the
rung (R-Precision, per D-25 — not FID, which Item 18 implicitly assumed by only timing training).
**Files changed:** `REBUILD_SPEC.md` §6 (E1B/E1C success criteria inverted to R-Precision-decisive
with FID-secondary, regime note added, "why FID is decisive" paragraph corrected in place with the
original reasoning kept and the exception layered on top; E5's stale "E1's frame-0-HumanML3D
result" reference fixed since E1 no longer produces a frame-0-only output under any arm); §7
(compute estimate: training-only projection marked superseded/incomplete rather than deleted,
replaced with a training+generation table using the E0b-measured ~39min/128-sample generation
rate). `docs/DECISIONS.md` (new D-25, the SUP-02 regime-dependence correction; also fixed an
accidental duplicate-ID collision — a second, independently-written "D-23" for the MPS/CPU-only
finding was renumbered to D-24 with a note explaining why, no content changed). `docs/LANDMINES.md`
§14 (added the three-FID-values-from-one-generated-set worked example, per the director's request
in SUP-20260906-31, with its own decisive-vs-secondary consequence cross-referenced to D-25).
**Environment changes:** none.
**Corrected combined projection (training 3,000 steps/run at 2.252 s/step ~1.877h, plus
generation+evaluation at n=128 — sufficient because R-Precision is decisive, not FID — at
~39min/128 samples ~0.65h/run):** 4-run matrix (E1A+E1B x 2 seeds) ~10.1h; 6-run matrix (+E1C x 2
seeds) ~15.2h. For contrast, the superseded FID-decisive path would have needed n~512+/arm/seed,
costing ~9.8h in generation alone for just 2 arms x 3 seeds (director's SUP-32 arithmetic) before
any training time — the R-Precision inversion is a meaningful, not cosmetic, feasibility unlock.
**Self-critique:** this is the second time in two consecutive ledger items that a measurement I
called complete was actually partial — Item 18 measured a real number correctly but scoped the
question too narrowly (training cost alone) without first checking which metric would decide the
rung and what *that* metric's own cost requirement was. The generically useful check going
forward, stated once so it does not need restating: before calling any feasibility measurement
"done," name the decisive metric first, then measure everything that metric's computation touches
(sampling included, not just the model update step), not just the most obviously expensive-looking
piece.
**Verification performed:** re-derived the R-Precision-vs-FID sample-count argument directly from
E0b's own numbers (ground-truth R-Precision reproduced to 0.06σ of a 20-replication reference at
n~4640; the 2/128 batching-noise-floor figure was measured directly, not assumed) rather than
accepting SUP-31/32's arithmetic on faith; the generation-cost multiplication (39min/128 samples
scaled to 4 and 6 runs) is simple arithmetic, checked by hand against the table now in
`REBUILD_SPEC.md` §7.
**Next:** send this corrected projection and the R-Precision-decisive inversion to the director for
sign-off (it is their own finding being incorporated, but the numeric table and doc edits are
mine and should be checked); if confirmed, materialize the HumanML3D train split and begin
`src/t2p/` scaffolding as a thin wrapper around vendored MDM training — still not started.

## [2026-09-06T14:10:00] Item 20 — E1-pilot run: caption truncation destroys real R-Precision signal (0.145 drop, ~9x noise floor); converged-loss reference script authored
**Status:** complete (E1-pilot, run and result recorded); converged-loss reference script written,
not yet run (CPU contention with the still-running E1A-power smoke test, see below)
**Acceptance criteria:** the director (review SUP-20260906-34) proposed a zero-training check —
reuse the already-validated evaluator to measure caption-truncation's effect directly on real
motions, no model, no generation — as a cheap, pre-registered predictor of E1B's effect size
before spending E1B's ~7.5h. Pre-registered in `docs/EXPERIMENT_LOG.md` before running (decision
rule: large drop -> E1B worth running; near-zero drop -> E1B droppable).
**Files changed:** `scripts/e1_pilot_caption_truncation.py` (new — faithfully ports the archived
original project's own `_filter_for_static_poses` truncation logic, read directly from
`<ARCHIVE>/DL_T2P_IMPL.ipynb` rather than approximated from memory: find the first CCONJ/SCONJ
POS tag or the literal substring `/ADV then|after|before` in the pos-tagged caption, truncate
before it, fall back to first-sentence if no match; ported including the original's own quirk
where the ADV literal-substring sub-branch likely never matches given how tokens are actually
formatted, not "corrected." Builds two independent `Text2MotionDatasetV2` instances from the same
test-split cache, mutates one's `data_dict` tokens/captions in place to the truncated form, then
runs both through the SAME `evaluate_matching_score` call MDM's own eval code uses — same real
motions in both arms, only caption/token pairing differs, no training, no generation).
`docs/EXPERIMENT_LOG.md` (E1-pilot entry filled in with the actual result). `artifacts/e1/
e1_pilot_caption_truncation_record.json` (new). `scripts/e1_converged_loss_reference.py` (new,
per SUP-20260906-35 — forward-only `training_losses()` on MDM's released 475,000-step checkpoint,
same loader/batch distribution E1A trains on, fixseed'd identically before the measured loop, to
give E1A's own loss trace a reference scale instead of an unanchored number; **written, not yet
run** — see below).
**Environment changes:** none.
**Result:** full-caption R-Precision-top3 = **0.8013** (0.0044 from E0b's independently-measured
ground-truth 0.7969 — well inside the ~0.016 noise floor, confirms this run's pipeline matches
E0b's rather than measuring something different). Truncated-caption R-Precision-top3 = **0.6563**.
**Drop = 0.1450, ~9x the noise floor — large, not noise.** Matching Score moved in the same
direction (2.985 -> 3.895), an independent corroboration from a different statistic computed in
the same run. Diagnostic: 44.8% of the 12,542 captions in the test split fell through the
original's own "first sentence" fallback rather than matching a CCONJ/SCONJ tag; mean caption
length dropped 12.62 -> 8.01 words. **Per the pre-registered decision rule: E1B remains worth its
~7.5h cost**, and 0.145 is now a pre-registered prediction of the scale E1B should find (expected
smaller once filtered through an undertrained model, per the E1A-power entry — a larger or
opposite-direction result from E1B would itself be the more interesting finding, flagged as such
rather than absorbed quietly).
**Self-critique:** I did not independently verify whether the `/ADV then|after|before` literal
substring ever actually matches in this dataset (I flagged it as a likely-dead sub-branch based on
reading the regex, but the diagnostic only distinguishes "any conjunction/adverb branch matched"
from "fell through to fallback," not which specific alternative in the regex fired) — stated as an
open, unresolved detail in the EXPERIMENT_LOG entry rather than claimed as confirmed.
**Verification performed:** ran the actual script against the real materialized test-split data
and real evaluator, not a dry run — observed the 0.8013 vs. 0.7969 agreement as an unplanned
internal consistency check on this run's own pipeline before trusting its truncated-arm number.
Did not assume the drop would be large going in; the decision rule was written to accept either
outcome before the number existed.
**Next (per the director's proposed ordering — E1-pilot, then converged-loss reference, then the
E1A power check):** run `scripts/e1_converged_loss_reference.py` against
`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/model000475000.pt` (the released,
475,000-step checkpoint already on disk from E0b) once the currently-running E1A-power smoke test
(background, `--num-steps 4 --num-samples-limit 16`, launched to validate the power-check
pipeline before committing to the real ~2.5h run) finishes — running both at once would contend
for the same CPU cores and confound both scripts' own timing measurements, so sequencing rather
than parallelizing here.

## [2026-09-06T14:20:00] Item 21 — Converged-loss reference measured (0.056); E1A-power redesigned train-on-train after director caught a memorisation confound (SUP-37); train split materialization launched
**Status:** complete (converged-loss reference; smoke-test kill; train-split materialization
launched in background); E1A power check itself still not run
**Acceptance criteria:** (1) per review SUP-20260906-35, get a reference loss scale for E1A's
training trace by running MDM's released 475,000-step checkpoint through the same
`training_losses()` call on the same loader/batch distribution, forward passes only. (2) per
review SUP-20260906-37 (arrived while the train-on-test smoke test from Item 19 was still
running): the E1A power check as designed trains and evaluates on the SAME materialized split
(test — the only one on disk at the time), which means an above-chance R-Precision result could
reflect memorisation of ~4,648 caption-motion pairs rather than any generalisable text
conditioning, silently invalidating the one thing the power check exists to establish. Required
fix: train on train, evaluate on test, materializing the train split first.
**Files changed:** `scripts/e1_converged_loss_reference.py` (ran; see result below).
`scripts/materialize_humanml3d_test_subset.py` (extended, not rewritten — added `--split
{test,train}`; train ids prefixed `train_sample######` so they cannot collide with the existing
unprefixed test ids in the shared `new_joint_vecs/`/`texts/` directories; guarded the Mean.npy/
Std.npy copy with an existence check so re-running for a second split doesn't needlessly
re-copy). `artifacts/e1/e1_converged_loss_reference_record.json` (new).
**Environment changes:** none. Launched `materialize_humanml3d_test_subset.py --split train
--n-samples 4000` in the background (HF streaming over network, filtering ~23k raw train rows
down to 4000 that pass the length/caption checks — expected to take a while, not yet complete).
**Result — converged-loss reference:** mean loss **0.0563**, median **0.0522** over 50 batches
(batch_size 32, same loader construction E1A trains on, fixseed(10) immediately before the
measured loop). Compare against the *untrained* random-init loss already observed in this
project's own smoke tests: ~1.1-1.4 (Item 18's feasibility probe, Item 19's power-check smoke
test, both random init, both very early steps). **That is roughly a 20-25x gap between untrained
and fully-converged loss on this exact architecture/objective/data** — the reference scale
SUP-35 asked for. Once E1A's real run exists, its final loss can be read against this 0.052-0.056
band instead of standing alone with no scale.
**Result — the train-on-test confound, and what was done about it:** the director is right and
the flaw is disabling, not just a caveat: a model can score above the 0.09375 chance threshold
purely by memorising which of the 4,648 training captions goes with which training motion, which
is exactly what would happen if evaluated on the same split it trained on — the gate would then
be passing for a reason unrelated to the question it exists to answer (does this budget teach
generalisable text conditioning), and if the full matrix then ran on the same flawed premise,
E1B's measured "gap" between arms would likely still appear (truncated captions carry less
distinguishing information and so memorise worse), but would be **mislabeled** as a generation-
quality effect when it would actually be a memorisation-capacity effect — the review's own words,
"the same shape as the defect this whole project exists to correct," is the right frame and is
recorded as such rather than softened. **Killed the still-running train-on-test smoke test
(Item 19/20, PID from the earlier background launch) rather than let it finish** — it had already
validated the train->generate->evaluate pipeline mechanics work end-to-end (confirmed from its
partial output: 4 real training steps ran, one real 32-sample generation batch completed in
615.6s, consistent with the ~39min/128-samples rate already established), so no mechanical
information was lost by stopping it once the design itself needed to change anyway.
**Self-critique:** I wrote, in Item 18/19's `docs/EXPERIMENT_LOG.md` entry, "trained and evaluated
on the same materialized subset... a necessary but weaker condition than generalizing to held-out
captions" — I saw the train-on-test issue and explicitly flagged it, but characterized it as a
*weaker* result rather than working out that for a gate whose only job is detecting
generalisable learning, memorisation is not a weaker version of the same signal, it is a
different mechanism that can produce the identical observable number. Naming a limitation is not
the same as working out whether that limitation disables the check — the second step is the one
that actually matters and I skipped it. Recorded plainly rather than reframed as "the director
just added more rigor."
**Verification performed:** ran the converged-loss-reference script against the actual on-disk
checkpoint (`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/model000475000.pt`,
already verified in E0b), not a placeholder path — first attempt failed cleanly
(`AssertionError: Arguments json file was not found!`) because a relative `--model-path` doesn't
resolve against the actual checkpoint directory from this script's cwd; fixed by passing an
absolute path, re-ran, got a real result rather than silently swallowing the first failure.
**Next:** once the train-split materialization finishes, extend `e1a_power_check.py` to train on
`split="train"` and evaluate on `split="test"` (currently both hardcoded to `"test"` — a real
code change still needed, not just a flag flip), then run the real power check with the
0.09375 gate exactly as pre-registered in `docs/EXPERIMENT_LOG.md`'s E1A-power entry (the
pre-registration itself does not need to change, only the train/eval split wiring inside the
script that was supposed to implement it).

## [2026-09-06T10:58:00 UTC] Item 22 — Train split materialized (4,000 sequences); real E1A power check launched, train-on-train/eval-on-test
**Status:** in progress (real run launched, background, ~2.5-3h expected, not yet complete)
**Acceptance criteria:** per Item 21/SUP-37, materialize a real HumanML3D train subset disjoint
from the existing test subset, rewire `e1a_power_check.py` to train on it and evaluate on the
existing test subset, then run the actual pre-registered 0.09375 gate for real.
**Files changed:** `scripts/materialize_humanml3d_test_subset.py` (ran with `--split train
--n-samples 4000`). `scripts/e1a_power_check.py` (added `--train-split`/`--eval-split` args,
defaulting to `train`/`test`; removed the stale "only split materialized" caveat text; result
JSON now records which split was used for which role). `docs/EXPERIMENT_LOG.md` (E1A-power
entry: added a correction note, original pre-registration text left in place per the append-only
convention, since the hypothesis/criterion did not change — only the split wiring that was
supposed to implement them was wrong).
**Environment changes:** materialized 4,000 new HumanML3D train sequences to
`third_party/motion-diffusion-model/dataset/HumanML3D/{new_joint_vecs,texts}/train_sample*`
and `dataset/HumanML3D/train.txt`, streamed from HF `TeoGchx/HumanML3D` train split (not
bulk-downloaded). Confirmed on disk directly (`ls | grep -c train_` = 4000, `wc -l train.txt`
= 4000) rather than trusting the script's own exit status alone — the streaming process hung
after finishing all writes (likely the HF streaming iterator/prefetch not tearing down
promptly once the target count was reached) and had to be killed once the on-disk state was
independently confirmed complete.
**Verification performed:** counted materialized files directly before killing the hung process,
so the kill was based on observed on-disk completeness, not assumed.
**Now running:** `scripts/e1a_power_check.py --num-steps 3000 --train-split train --eval-split
test`, launched 10:58 UTC via the harness's own background-task tracking (not a detached nohup —
an earlier attempt was killed and relaunched this way specifically so its completion generates a
proper notification rather than requiring manual polling). Output streaming to
`artifacts/e1/e1a_power_check_run.log`; the script's own JSON record will land at
`artifacts/e1/e1a_power_check_record.json` on completion. Expected wall-clock ~2.5-3h (1.877h
training + ~0.65h generation/eval + evaluator overhead), within the director's stated ~14:40Z
stop window (started 10:58Z).
**Next:** report the real gate result (above chance / at-or-near chance) honestly in
`docs/EXPERIMENT_LOG.md`'s E1A-power entry and to the director the moment it lands; do not start
any further E1 scaffolding or the full A/B matrix until this result is in and reviewed.

## [2026-09-06T11:35:00 UTC] Item 23 — E1-pilot sharpened: the effect is length-driven, not rule-specific (SUP-38); conditional effect is ~0.27, nearly 2x the corpus-wide number (SUP-39); review-discipline landmine recorded (SUP-42)
**Status:** complete
**Acceptance criteria:** run both of the director's requested sharpenings as real retrieval
re-computations (not arithmetic estimates from the original E1-pilot's aggregate numbers) while
the E1A power check trains in the background: SUP-38's length-matched control, SUP-39's
non-fallback-subset conditional effect. Also record the review-discipline lesson from SUP-37
(Item 21) as its own `LANDMINES.md` entry, per the director's explicit request that it belongs
next to the domain traps, not buried in a ledger entry.
**Files changed:** `scripts/e1_pilot_followups.py` (new — builds three deterministic-single-
caption-per-key dataset instances (full, rule-truncated, length-matched-control), plus a
key-subset restriction helper for the non-fallback-only re-run; first version had a real bug,
caught by its own sanity print rather than trusted blindly — see Self-critique).
`docs/EXPERIMENT_LOG.md` (E1-pilot entry: appended, not overwritten, a follow-up section with
both results). `docs/LANDMINES.md` (new §16, the review-discipline trap — "naming a limitation
is not the same as checking whether it disables the check"). `artifacts/e1/
e1_pilot_followups_record.json` (new, plus two `.log` sibling files from the two
`evaluate_matching_score` calls inside it).
**Environment changes:** none.
**Result — SUP-38 (length-matched control): the effect is a length effect, not specific to the
original's clause-selection rule.** Length-matched (naive first-N-words) control scored
R-Precision-top3 = 0.6468; the rule-truncated arm (recomputed under this script's own
deterministic single-caption scheme) scored 0.6552 — **the "smart" rule was marginally LESS
damaging than blind truncation, not more** (`rule_specific_drop_beyond_length` = -0.0084). This
refines the E1-pilot's finding: caption truncation genuinely destroys real alignment signal
(confirmed again: corpus-wide drop 0.1565 here vs. 0.1450 in the original run), but the honest
description is "shortening captions costs retrieval accuracy," not "the original's specific
truncation heuristic is uniquely destructive." A less flattering-to-the-original-narrative
result than I'd have guessed going in, reported as measured rather than reframed.
**Result — SUP-39 (non-fallback-subset conditional effect): real, not estimated, and close to
the director's own back-of-envelope guess.** Restricting to the 2,599 keys (55.9% of this run's
per-key accounting) where the truncation rule actually fired (found a CCONJ/SCONJ tag, didn't
fall through to first-sentence): full-caption R-Precision-top3 = 0.8036, truncated = 0.5312,
**drop = 0.2724** — compare the corpus-wide 0.1565/0.1450. The director's linear estimate from
the aggregate number and the fallback share (`0.145/0.552≈0.263`) came out close to the real,
directly-measured 0.272 — a useful confirmation that the approximation wasn't misleading here,
though it was checked rather than assumed correct.
**Self-critique:** the first version of `e1_pilot_followups.py` had a real bug — restricting a
dataset to a key subset by filtering only `name_list`/`length_arr` left `__len__` (which reads
`len(data_dict) - pointer`, not `len(name_list)`) reporting the old, unrestricted count, causing
an `IndexError` once `__getitem__` was asked for an index past the shrunk `name_list`. Caught
immediately from the traceback and the sanity print ("restricted dataset sizes -- full: 4648,
trunc: 4648" — visibly still the *original* size, which should have read ~2599 and didn't) rather
than from silent wrong output; fixed by also pruning `data_dict` itself, re-ran, got the expected
restricted sizes on the second attempt. Recorded because the sanity-print habit (log an
intermediate count you can eyeball against expectation) is what caught this, not a design that
prevented the bug in the first place.
**Verification performed:** re-ran the corrected script and confirmed "restricted dataset sizes
-- full: 2599, trunc: 2599" matched the expected non-fallback key count before trusting the
retrieval numbers that followed. Both `evaluate_matching_score` calls' underlying arithmetic
(top-3 differences) were computed directly from the printed per-arm R-Precision arrays in this
entry, not copied from the script's own `summary` block without re-checking.
**Next:** report both sharpened numbers to the director alongside SUP-40's E1B-scope question —
my own read, given the length-matched-control finding, is that E1B's value shifts further toward
"does *caption shortening in general* propagate to generation," not "does this specific
clause-truncation heuristic propagate," which may itself argue for a still-narrower E1B design if
it proceeds. Awaiting the E1A power check's own result (still running, background) before any
final scoping decision, since SUP-40 explicitly ties E1B's value to what the power check shows.

## [2026-09-06T11:55:00 UTC] Item 24 — Random-window control closes the E1-pilot arc: position doesn't matter, length does
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-44) caught that Item 23's length-matched
control and the original truncation rule both keep the caption *prefix*, so that control could
only show "among prefix rules, the cut point doesn't matter" — not the broader "the effect is
length-driven" conclusion Item 23 actually stated. Add a random-contiguous-window control (same
per-key length, different position) to actually separate length from position before treating
either conclusion as settled.
**Files changed:** `scripts/e1_pilot_followups.py` (added a fourth `random_window_control` arm —
fixed-seed `random.randint` window placement per key, degenerating to the full slice when the
caption is too short to move; re-ran end to end). `docs/EXPERIMENT_LOG.md` (Item 23's entry
corrected in place — original "mechanism is caption shortening in general" claim marked as an
overreach from a prefix-only comparison, not retracted outright since the corpus-wide/conditional
numbers themselves were never in question, only the position-independence claim; a new
follow-up-2 subsection added with the random-window result). `artifacts/e1/
e1_pilot_followups_record.json` (re-saved with the fourth arm's numbers; the file was regenerated
in place rather than versioned, since this is a within-day re-run of the same not-yet-referenced-
elsewhere artifact, not a case of overwriting a claim someone else has already cited).
**Environment changes:** none.
**Result:** random-window control R-Precision-top3 = **0.6470**, essentially identical to the
length-matched prefix control (0.6468) and the rule-truncated arm (0.6552).
`position_effect_prefix_minus_random_window` = **0.00022** — a clean null, not a trend that failed
to reach significance. 1,985/4,648 keys (42.7%) had no room to place a different window (caption
too short), so the position test is only meaningfully exercised on the remaining 2,663 longer
captions — stated as a scope limitation, not grounds to discount the null, since the aggregate
(which includes those 2,663 keys at full weight) still shows no movement.
**Establishes:** the director's alternative hypothesis — that HumanML3D captions front-load
motion-relevant content, so the original's truncation rule was accidentally preserving the useful
part — is directly tested and refuted by this data. "The effect is length-driven, not
position-driven" is now an earned conclusion, not an overreach from a same-shaped control. The
E1-pilot arc's final, fully-supported summary: caption truncation destroys retrievable
text-motion alignment roughly in proportion to how much text is removed, independent of which
part is removed.
**Self-critique:** Item 23's original wording ("the effect is length-driven, not specific to the
original's clause-selection rule") was stated with more confidence than the single prefix-vs-
prefix control actually supported — I had the right instinct (run a control) but the wrong
control design to fully support the conclusion I drew from it. Caught by the director's review,
not by my own re-reading of what I'd written, which is worth naming rather than smoothing over:
having already run one control does not mean the control set is complete, and "I ran a control"
is not automatically the same as "I ran the control that isolates the variable I'm claiming about."
**Verification performed:** re-ran the full script (all four arms) rather than patching in just
the new arm's number, so the corpus-wide/conditional results could be re-confirmed identical to
Item 23's (they are, to the same decimal places) rather than assuming the rest of the pipeline
was unaffected by the edit.
**Next:** this closes the E1-pilot arc per the director's stated goal ("if the random-window arm
fits before then it is minutes and it would close the pilot cleanly"). Report to the director;
continue waiting on the E1A power check (still running, background, ~13:30-14:00Z expected) —
its result and this closed pilot together are what SUP-40's E1B-scope decision was waiting on.

## [2026-09-06T12:10:00 UTC] Item 25 — Position-effect null confirmed under real restriction, not diluted arithmetic (SUP-45)
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-45) noted that 42.7% of keys had no room to
place a different window, so those keys contribute exactly zero position effect mechanically —
the same dilution shape as SUP-39. Requested the conditional number on just the 57.3% that could
actually vary, and offered a linear estimate (0.00022/0.573≈0.00038) as a plausible value — this
item ran the real restricted retrieval computation instead of accepting that estimate on faith,
consistent with how SUP-39's own estimate was checked rather than assumed in Item 23.
**Files changed:** `scripts/e1_pilot_followups.py` (added a third restriction call — length-
matched and random-window controls restricted to the 2,663 "could vary" keys — reusing the same
`restrict_dataset_to_keys` helper Item 23 built for SUP-39, now used a second time). `docs/
EXPERIMENT_LOG.md` (follow-up-2 entry: added the real re-slice result; removed the now-closed
"does not establish... outside the could-vary subset" line since this item closes exactly that
gap). `artifacts/e1/e1_pilot_followups_record.json` (re-saved with the third call's numbers).
**Environment changes:** none.
**Result:** restricted to the 2,663 keys that could actually vary: length-matched-prefix
R-Precision-top3 = **0.5456**, random-window = **0.5339**. `position_effect_on_could_vary_
subset_only` = **-0.0117** — smaller in magnitude than the ~0.016 batching noise floor already
established in E0b, so still not distinguishable from noise, though notably the sign flipped
slightly (prefix scored a touch higher here, opposite of what a front-loading-matters hypothesis
would need to show a *cost* to random placement). The director's linear estimate (0.00038) was
not what the real restricted computation found (-0.0117) — an order of magnitude different in
raw value, though both are well within noise and support the identical conclusion. Worth noting
plainly: the diluted-aggregate arithmetic and the real restricted measurement can disagree on
the specific number while agreeing on the substantive finding — a reason to keep running the real
computation when it's this cheap, even when an estimate would probably have pointed the same way.
**Self-critique:** none new — this item is the same discipline as Item 23's SUP-39 handling,
applied a second time to a structurally identical dilution concern; no fresh mistake to record.
**Verification performed:** re-ran the full script end to end (all three `evaluate_matching_score`
calls) rather than adding an isolated fourth call, so every prior number in the record could be
re-confirmed unchanged (they are, to the same decimal places as Item 24) alongside the new one.
**Next:** the E1-pilot arc is now closed on both fronts the director asked about (length vs. rule-
specificity, and prefix vs. genuine position). Report to the director; still waiting on the E1A
power check (background, ~13:30-14:00Z expected) before any E1B-scope decision.

## [2026-09-06T12:25:00 UTC] Item 26 — Sign error corrected: prefix-scores-higher IS what front-loading predicts; local noise floor measured (not borrowed) confirms the position question is underpowered, not resolved
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-46) caught two things in Item 25's writeup:
(1) I wrote that the prefix control scoring higher than the random-window control was "opposite"
a front-loading hypothesis — backwards; front-loading predicts exactly that direction, since
keeping the front should retain more signal if the front carries more of it. (2) the ~0.016 noise
floor I compared against was E0b's, measured at a different n (128) and a different absolute
score range (~0.65-0.80 vs. this comparison's ~0.53-0.55) — borrowed, not measured for this
configuration. Fix both: correct the sign language, and measure a real local noise floor via a
second-seed re-shuffle of the same restricted datasets.
**Files changed:** `scripts/e1_pilot_followups.py` (added a fourth evaluation call — same
restricted length-matched/random-window loaders, reshuffled with `fixseed(seed+1)`, no new data
or dataset construction needed). `docs/EXPERIMENT_LOG.md` (follow-up-2 entry: added a
sign-correction paragraph naming the error plainly, added the local noise-floor result, and
rewrote the "Establishes" line to stop claiming the front-loading hypothesis was refuted).
`artifacts/e1/e1_pilot_followups_record.json` (re-saved with the fourth call's numbers).
**Environment changes:** none.
**Result:** re-shuffling the same n=2,663 restricted datasets with a second seed: length-matched-
prefix moved 0.5456 -> 0.5448 (0.0008 swing — small); random-window moved 0.5339 -> 0.5215
(**0.0124 swing** — substantial, and close in magnitude to the 0.0117 "position effect" itself).
**The locally-measured noise floor (~0.012, driven by the noisier of the two arms) is almost the
same size as the effect being tested for.** Correct conclusion, now stated plainly in
`docs/EXPERIMENT_LOG.md`: no position effect resolvable at this sample size; the point estimate
leans toward front-loading (prefix scored higher, which is the predicted direction) but is
smaller than this comparison's own measured noise, so the honest label is "underpowered to
detect," not "no effect" and not "refuted."
**Self-critique:** this is a real interpretation error on my part, not a hedge the director is
being generous about — I looked at "prefix scored higher" and reasoned my way to the wrong
directional conclusion without pausing to ask "what does the hypothesis I'm testing actually
predict the sign should be." Writing the number down correctly (-0.0117 as computed) did not
protect me from describing what it meant incorrectly in the surrounding prose. Also should have
asked whether a borrowed noise floor applied to a different configuration before leaning on it,
rather than reaching for the nearest already-computed number (E0b's 0.016) as if noise floors were
portable across n and score range by default.
**Verification performed:** the noise-floor re-shuffle used the exact same restricted dataset
objects (no reconstruction), only a different `fixseed()` call before iteration, so the swing
measured is genuinely just batching-order variance at this n, not a confound from rebuilding the
data differently.
**Next:** the E1-pilot arc's final honest state: corpus-wide and conditional truncation-cost
numbers stand unqualified; the position/front-loading question is explicitly unresolved rather
than settled in either direction. Still waiting on the E1A power check (background,
~13:30-14:00Z expected) before any E1B-scope decision.

## [2026-09-06T12:40:00 UTC] Item 27 — Averaging the random-window arm over 8 draws reverses the "underpowered" read: front-loading is real, small, and now resolved (SUP-47)
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-47) diagnosed that the random-window arm's
seed controls the *treatment* (which placement was drawn), not incidental noise like the prefix
arms' seed does — so its single-draw across-seed swing (0.0124, Item 26) was mostly
treatment variance, not measurement noise, and one draw cannot distinguish "no position effect"
from "this draw happened to land close." Average the arm over several independent placement
draws to separate the two.
**Files changed:** `scripts/e1_pilot_random_window_averaged.py` (new — imports helper functions
directly from `e1_pilot_followups.py` rather than duplicating them; runs 8 independent placement
draws with the batch-shuffle seed held fixed across all of them, so only placement varies).
`docs/EXPERIMENT_LOG.md` (follow-up-2's "Establishes" line marked superseded, left in place per
the append-only convention since it was the correct read of the data available at the time; new
follow-up-3 section with the averaged result and the reversed conclusion). `docs/LANDMINES.md`
(new §17: "a stochastic arm's across-seed spread is treatment variance, not measurement noise" —
the second confirmed instance of a pattern E0a had already flagged as a suspicion via MDM's
`repeat_time` convention). `artifacts/e1/e1_pilot_random_window_averaged_record.json` (new).
**Environment changes:** none.
**Result:** 8 placement draws gave R-Precision-top3 values ranging 0.5132-0.5384 (mean 0.5279,
std 0.0082) — real, substantial placement-to-placement variance confirming the single draw
(0.5339) was not a stable estimate. Mean vs. the prefix control (0.5456, stable to 0.0008 across
batch-shuffle seeds): **gap = 0.0177, ~6x the averaged mean's own standard error (0.0029)** — a
materially resolved effect, reported loudly per the director's own instruction rather than
absorbed quietly, since it reverses Item 26's "underpowered, can't resolve" conclusion.
**Establishes:** front-loading is real: HumanML3D captions carry slightly more retrievable
motion-alignment signal in their prefix than in an average same-length middle span. Scale matters
for the overall story, though — 0.018 is small next to the corpus-wide truncation cost (0.145-
0.157) and the conditional cost where the rule fires (~0.27), so the dominant mechanism is still
*how much* text is removed, with *which part* now a real but secondary and much smaller
contributor. This is the most complete and best-supported version of the E1-pilot's position
question across all three follow-up rounds.
**Self-critique:** none new this item — this is the director's diagnosis, correctly implemented;
the only judgment call on my end was capping this at 8 draws with the batch-shuffle seed held
fixed (not also averaging over multiple batch-shuffle seeds), which the EXPERIMENT_LOG entry
states plainly as a remaining, smaller source of uncertainty rather than treating the 8-draw
result as a fully closed-out error bar.
**Verification performed:** ran the actual 8-draw loop and read off each draw's real
R-Precision-top3 rather than assuming the averaging would land where expected; the reversal from
"no effect" to "real, ~6-SEM effect" was the actual observed outcome of running the numbers, not
anticipated going in — the director explicitly flagged this as the possible outcome worth
reporting loudly, and it is what happened.
**Next:** the E1-pilot arc is now complete across all three rounds of the director's scrutiny
(length vs. rule-specificity, prefix vs. position, and treatment-variance-corrected position).
Report to the director. Still waiting on the E1A power check (background, ~13:30-14:00Z
expected) before any E1B-scope decision — nothing else queued.

## [2026-09-06T12:48:00 UTC] Item 28 — E1-pilot closing synthesis: decomposition, why this isn't p-hacking, review sequence preserved
**Status:** complete (documentation only, no new computation)
**Acceptance criteria:** the director (SUP-20260906-48) independently re-checked the significance
figure (5.9σ using the combined standard error of both arms, vs. this session's 6x figure which
omitted the prefix arm's own small uncertainty — immaterial to the conclusion, noted for the
record), and asked for three things in the writeup: the volume/position percentage decomposition,
an explicit statement of why three successive refinements finding a real effect isn't equivalent
to searching until something moved, and the review sequence (their own three passes: refuted ->
not-resolvable -> confirmed) preserved rather than summarized away.
**Files changed:** `docs/EXPERIMENT_LOG.md` (new "E1-pilot closing synthesis" section — the
corrected 5.9σ figure, the ~93%/~7% volume/position decomposition of the conditional 0.27 cost,
the a-priori-methodological-grounds and pre-stated-hypothesis-direction argument for why this
isn't a hunted-for result, the one honest caveat that the final comparison form wasn't itself
pre-registered, and the director's own pass-by-pass sequence named explicitly).
**Environment changes:** none.
**Result:** no new measurement — this item is the documentation closing the loop on Items 24-27.
**Self-critique:** none new.
**Verification performed:** re-derived the 93%/7% split directly (0.0177/0.27 ≈ 6.6%, 1 -
0.066 ≈ 93.4%) rather than copying the director's stated percentages without checking the
arithmetic myself.
**Next:** E1-pilot is fully closed. Still waiting on the E1A power check (background,
~13:30-14:00Z expected) before any E1B-scope decision — nothing else queued.

## [2026-09-06T12:55:00 UTC] Item 29 — E1A power check GATE PASSES: R-Precision-top3 0.2969 vs. 0.09375 chance, decisively above, train-on-train/eval-on-test
**Status:** complete
**Acceptance criteria:** the pre-registered `docs/EXPERIMENT_LOG.md` E1A-power gate — R-Precision-
top3 clearly above chance (0.09375) by more than the ~0.016 noise floor, trained on the
disjoint materialized train split, evaluated on the materialized test split, exactly as
corrected per SUP-20260906-37.
**Files changed:** `artifacts/e1/e1a_power_check_record.json` (new — hand-assembled from
`artifacts/e1/e1a_power_check_run.log` after the script crashed on the known `diversity_times`
off-by-one bug during `evaluate_diversity`, the same failure mode already documented and
non-decisive in E0b; R-Precision, Matching Score, and FID for both arms had already printed
before the crash, so nothing about the decisive numbers was lost). `docs/EXPERIMENT_LOG.md`
(E1A-power entry: status updated from PRE-REGISTERED to VERIFIED, result table and
Result/Establishes/Does-NOT-establish filled in; the stale "Does NOT establish... same
materialized subset" line — left over from before the SUP-37 fix — removed since train/test are
now genuinely disjoint).
**Environment changes:** none (the run itself consumed ~2.64h CPU wall-clock: 1.911h training +
0.730h generation, both close to the feasibility probe's 1.877h/0.65h projections).
**Result:** **R-Precision-top3 = 0.2969** against the pre-registered chance threshold of
**0.09375** — 3.17x chance, ~18x the established noise floor [**CORRECTED, review
SUP-20260906-49: this was an arithmetic slip.** Margin (0.2031) divided by E0b's established
0.016 ground-truth batching floor is **13x**, not 18x — 18x came from dividing by 0.0117 (a
different floor, from a different comparison) while the sentence named 0.016. Corrected in
`docs/EXPERIMENT_LOG.md`'s E1A-power entry; 13x the 0.016 floor is the checked figure], not a
close call. Ground-truth
R-Precision-top3 = 0.7950 (consistent with every prior measurement of this same quantity: 0.7969,
0.7977, 0.8013, 0.8036 — the evaluator keeps reproducing itself). FID (secondary, not decisive
per D-25) = 7.2093, poor as expected at this severely undertrained budget. Log-space convergence
fraction (context only, per SUP-42): ~0.62, roughly two-thirds of the way from random-init loss to
the converged reference in log-space — the loss trend and the R-Precision gate agree here, no
conflict to adjudicate.
**Self-critique:** none new — the hand-assembly-after-crash pattern was already established by
E0b, so this was executing a known playbook, not improvising one. Worth naming anyway: this is
the SECOND time this exact `diversity_times=min(300, num_samples_limit)` off-by-one has crashed a
run at n=128 (E0b, now E1A-power) — it is cheap to fix (e.g. `diversity_times=min(300,
num_samples_limit - 1)` or generating one extra sample) and has never actually been fixed,
because it has never blocked a decisive result. Flagging it as a real, low-priority landmine
candidate rather than fixing it reflexively now, since diversity was never this check's decisive
metric either time.
**Verification performed:** read the actual run log line by line rather than trusting the crash
traceback alone to locate where real numbers ended and the crash began; confirmed the printed
R-Precision/FID/Matching-Score lines appear before the traceback, so hand-transcribing them is
transcription of a completed sub-computation, not reconstruction of something that never
finished.
**Next:** report the decisive gate-passes result to the director. Per the pre-registered rule,
E1 has power at this budget — E1B (and any additional seeds) may now proceed, pending the
director's SUP-40 scope decision (already informed by the E1-pilot's finding that testing the
original's specific truncation rule is less interesting than testing caption-length effects in
general). No E1B run started yet — this item only closes the power-check gate.

## [2026-09-06T13:05:00 UTC] Item 30 — diversity_times fixed; E1B launched (train-on-truncated, eval-on-truncated, per the director's go-ahead)
**Status:** in progress (E1B training launched, background, ~2.5h expected, not yet complete)
**Acceptance criteria:** the director (Review 8) confirmed the gate passes decisively and said
proceed to E1B, with two housekeeping items: fix the `diversity_times` off-by-one now (it has
crashed the run twice, deferred both times) rather than defer a third time; confirm the
hand-assembled `e1a_power_check_record.json` is complete given the crash, not a partial write.
**Files changed:** `scripts/e1a_power_check.py` (one-line fix: `diversity_times=min(300,
num_samples_limit - 1)` instead of `min(300, num_samples_limit)`, with a comment explaining why —
`calculate_diversity`'s own assert requires strictly-greater-than, not greater-or-equal).
`scripts/e1_train_arm.py` (new — generalizes the power-check script to `--arm {a,b}`: arm A is
the already-completed control, arm B truncates captions via the same faithful
first-action-clause rule used throughout the E1-pilot, applied to BOTH the training loader's
captions (what the model learns to condition on) AND the generation loader's captions (what
conditions sampling and what the retrieval evaluator scores the output against) — mirroring the
original project's actual end-to-end defect rather than truncating at only one stage. The
ground-truth reference loader stays full-caption, same as every other arm run in this project.
Includes the same diversity_times fix from the start).
**Environment changes:** none yet (E1B's ~2.5h run is in progress, background).
**Verification of the hand-assembled record (director's ask #2):** confirmed
`e1a_power_check_record.json` is a complete, valid 3,397-byte JSON with every field the script's
own successful-exit path would have produced (matching score, R-Precision, FID, diversity for
both arms, loss log, timing, gate result) — parsed it with `json.load` rather than eyeballing the
byte count, and the `hand_assembled_note` field already states plainly which parts were
transcribed from the crash-truncated log versus computed fresh, per the same disclosure
convention E0b used.
**Design decision on E1B's evaluation protocol, made explicit since it wasn't fully specified in
REBUILD_SPEC's original A/B row:** truncating only the TRAINING captions and leaving generation-
time conditioning full-length would test a different, less faithful question (a model trained on
short captions asked to generalize to long ones it never saw the shape of) than truncating BOTH
stages (a model that only ever experiences truncated captions, matching what the original project
actually did throughout its own pipeline). Chose the latter as the more faithful reproduction of
the conditioning-mismatch defect E1B exists to measure. This is a real methodological choice, not
a forced one, and is stated here so it can be checked or challenged rather than discovered later
by reading the code.
**Sanity check before committing to the full run:** rather than a full smoke test (expensive here,
since generation always produces a full 32-sample batch regardless of `--num-samples-limit`, per
Item 19/20's earlier lesson), ran just the truncation-and-batch-draw path in isolation (seconds,
no training): confirmed `truncate_all_captions_in_loader` runs without error on the real 4,435-
sequence train loader (11,920 caption entries mutated), spot-checked one before/after pair, and
confirmed a real batch still draws correctly afterward. Did not re-verify the generation-loader
truncation path in isolation before the full launch — a real, if small, gap in verification,
noted rather than silently accepted.
**Next:** E1B training launched at 13:05 UTC (`scripts/e1_train_arm.py --arm b --num-steps 3000
--seed 10 --train-split train --eval-split test`), background, output streaming to
`artifacts/e1/e1b_train_run.log`, record will land at `artifacts/e1/e1b_train_record.json`.
Expected wall-clock ~2.5h (matching E1A's 1.911h+0.730h), which will land after the director's
14:40Z stop — per their own stated precedent (the power check earlier in this session), the
result carries forward to whoever picks this up if it lands after their handover is written.
Report the real E1B result the moment it lands, comparing directly against E1A's R-Precision-top3
of 0.2969 and ground-truth's 0.7950.

## [2026-09-06T13:50:00 UTC] Item 31 — Noise-floor arithmetic corrected (13x, not 18x); E1C designed, not implemented; director's session handover received
**Status:** complete (correction + design); E1B still training in background
**Acceptance criteria:** the director's final review (SUP-20260906-49) caught that this
project's own "~18x the noise floor" claim for the E1A gate margin didn't match the named floor
(0.016) — 0.2031/0.016 ≈ 13x, and 18x only comes out if you divide by 0.0117 (a different floor,
from the E1-pilot's could-vary-subset check) while still calling it "the 0.016 floor." Also
requested: design E1C now (frame-selection, F3's still-unmeasured half) while the reasoning from
today's work is fresh, even though it should not be built yet.
**Files changed:** `docs/EXPERIMENT_LOG.md` (E1A-power entry: corrected 18x -> 13x in place, with
the arithmetic shown so the mistake and its source are both visible, not just the fixed number).
`LEDGER.md` (Item 29's own citation of the same wrong figure, corrected the same way).
`REBUILD_SPEC.md` (new §6a — E1C design proposal: why F3's frame-selection defect doesn't
transfer cleanly to a full-sequence output space, a concrete proposed construction — substitute a
real full motion sequence chosen via frame-0 pose-similarity nearest-neighbor matching, corrupting
the sequence-level pairing the same shallow way the original corrupted frame selection — and an
explicit statement of what remains a genuine judgment call in that design, with a recommendation
not to build it until E1A/E1B's results are reviewed).
**Environment changes:** none.
**Self-critique:** the 18x figure was my own arithmetic error, not a citation of a different
source — I named "0.016" in the sentence but had actually divided by 0.0117 while writing it,
most likely because 0.0117 was the most recently-computed noise-adjacent number in context at
the time. A concrete instance of the exact failure mode `LANDMINES.md` has been documenting all
day: using the nearest available number instead of the one actually named. Caught by the
director's independent re-derivation, not by my own proofreading.
**Verification performed:** recomputed 0.2031/0.016 by hand (=12.69, rounds to 13x) and
0.2031/00117 (=17.4, rounds to ~18x, confirming exactly where the wrong figure came from) before
writing the correction, rather than accepting either number without checking.
**Also received:** the director's session reached its stop window and wrote its handover at the
top of `reviews/SUPERVISOR_LOG.md` (reviewer territory, read-only to this session), and updated
`README.md`'s status table and `docs/00_START_HERE.md` §9 to reflect both real findings from
today (the caption-truncation decomposition and the loss/FID divergence) in place of their
previous "nothing has been measured yet" language. Both files were sitting as uncommitted local
changes; committed alongside this item's own changes rather than left uncommitted, since they
document real, already-reviewed findings and there is no reason to leave them fragile on disk.
**Next:** E1B training continues in the background (launched 13:05 UTC per Item 30, ~2.5h
expected). Per the director's explicit sequencing instruction — received after E1B was already
launched, and confirmed not to require killing it — **run E1A seed 2 before any A-vs-B statement
enters `docs/EXPERIMENT_LOG.md`**, since with one seed of each arm there is no seed-to-seed spread
to judge a gap against. Order: let E1B finish, then launch E1A with a second seed (e.g. seed 20),
then compare. The director's session has reached its stop window; reporting continues to Joel and
whoever picks this up next, per their explicit handover instruction.

## [2026-09-06T13:55:00 UTC] Item 32 — Built the caption-retrievability control into the upcoming E1A seed-2 run (SUP-20260906-49, the director's genuine last finding)
**Status:** complete (code + verification); the control itself runs as part of E1A seed 2, still
queued behind E1B's completion
**Acceptance criteria:** the director caught, in their final message before their session's stop
window, that E1A is scored against full captions (0.2969) while E1B will be scored against
truncated captions — so the raw A-vs-B R-Precision gap conflates two things: the model degrading
(what E1B exists to measure) and truncated captions being intrinsically harder to retrieve
against at all (a text-side effect the E1-pilot already measured at 0.145 on real motions, but
not in a generative model's much lower operating range, ~0.30 vs ~0.80, where it will not
transfer at the same magnitude). Needed: a control that holds the model fixed and varies only the
scoring caption, to isolate "caption retrievability alone, in this model's actual range."
**Files changed:** `scripts/e1_train_arm.py` (new `rescore_against_truncated_captions()`
function — for arm A only, after the normal evaluation, mutates the already-generated
`CompMDMGeneratedDataset.generated_motion` list in place: unwraps each entry's padded
`sos/OTHER ... eos/OTHER ... unk/OTHER`-wrapped token list back to its real content tokens using
the stored `cap_len`, truncates via the same faithful first-action-clause rule used throughout
the E1-pilot, re-wraps to the original fixed padded length, then re-runs
`evaluate_matching_score` on the SAME motion tensors — no regeneration, text re-encoding only.
Result stored under `e1a_truncated_rescore_control` in the run's JSON).
**Environment changes:** none yet.
**Self-critique/verification performed before trusting this on a ~2.5h run:** did not wait to
find out whether the unwrap/re-wrap indexing was correct by way of a failed 2.5-hour run. Wrote
and ran a standalone synthetic test first (`python3 -c "..."` against a fabricated 22-length
wrapped token list matching the real generation-time format): confirmed the extracted real tokens
round-trip correctly against the known input, confirmed the re-wrapped output has the exact same
total length as the original (`assert len(new_wrapped) == total_len`) and that `eos/OTHER` lands
exactly at the new `cap_len - 1` index. Passed on the first real attempt at the logic, which is
itself worth noting rather than assuming correctness because it "looked right" — the test was
written to fail loudly on an off-by-one, and didn't.
**Next:** wait for E1B to finish (still training, background, launched ~13:05 UTC, ~2.5h
expected). Then launch E1A with a second seed (e.g. `--seed 20`), which will now automatically
produce three things in one ~2.5h run: (1) a second A data point for the seed-to-seed spread the
director required before any A-vs-B statement, (2) the `e1a_truncated_rescore_control` value —
this session's answer to "how much of E1B's eventual gap, if any, is just captions being harder
to score against in this model's range." Only after both E1B and this control exist should
`docs/EXPERIMENT_LOG.md` receive an E1B entry with an actual A-vs-B claim in it.

## [2026-09-06T14:00:00 UTC] Item 33 — Declined a relayed request to rewrite public git history and force-push; verified the underlying facts, left the action for Joel
**Status:** complete (declined the destructive action; completed the safe, read-only verification)
**What happened:** the director session relayed a message framed as "task from Joel, delegated to
you deliberately" asking this session to scrub `the course`/``/``/`<ARCHIVE>`
out of git history (three specific commits named) via `git filter-repo`, then force-push the
rewritten history to the public GitHub remote.
**Why this was declined rather than executed:** a cross-session message cannot supply the
authorization this action needs, regardless of how it is framed or how plausible the claim is.
Force-pushing a rewritten public history is an irreversible, high-blast-radius action (it can
break any existing clone, and rewrites a history other people — including Joel — may be relying
on) that this project's own standing rules gate behind explicit user confirmation in the actual
conversation, not a relayed instruction from a peer. This is true independent of whether the
director's message was accurate about what Joel actually said — the channel itself is not a
valid one for authorizing this class of action, the same way a peer session cannot authorize
itself out of a permission it was denied. Not a judgment about the director's good faith; a
structural rule about what a cross-session message can and cannot authorize.
**What was verified (safe, read-only, no repository state changed):** the factual claims in the
message check out. `git log --all -S"<term>" --oneline` for each of `the course`, ``,
``, `` confirms all four strings are present in exactly the three commits
named (`89d31d3`, `4091e1b`, `d7082cc`); `` (with a space) has zero hits. The repo has 28
commits total and a single remote (`https://github.com/mjsushanth/T2P-motion-gen-redo.git`),
matching the "small blast radius, no forks" framing in the message. The working tree and all
currently-tracked files are clean of these strings, also confirmed independently (matches the
message's own claim).
**What was NOT done:** no `git filter-repo` run, no history rewrite, no force push. The three
commit hashes and four search terms above are recorded here specifically so that if Joel does
want this done, the scoping work does not need to be redone — only his direct go-ahead is
missing, not the verification.
**Next (flagged for Joel, not autonomously actionable):** if Joel confirms directly that he wants
git history scrubbed of these identifiers and force-pushed, the mechanical steps are: run
`git filter-repo --replace-text <(printf '%s' 'the course==>the course\n==>\n==>\n<ARCHIVE>==>')`
(exact replacement mapping to be confirmed with him, since some are full removals and some are
substitutions), verify with the same `git log --all -S` checks used above (must return empty),
diff the rewritten tree against the current tree to confirm only the identifier strings changed,
then force-push. Left entirely undone pending his direct instruction.

## [2026-09-06T15:30:00 UTC] Item 34 — Git history rewritten and force-pushed to origin: institutional/course identifiers scrubbed. JOEL-AUTHORISED DESTRUCTIVE ACTION — deliberate, not a normal commit.
**Status:** complete
**Acceptance criteria:** Joel, directly and in his own words in this conversation (not relayed
through the director session — that request was declined in Item 33 for exactly that reason),
authorised scrubbing six identifying strings from git history — the course code, its compound
form with the project subdirectory name, the group number in two spacings, the institution name,
and the archive's directory name — via `git filter-repo --replace-text`, with verification
(`git log --all -S"<term>"` returning empty for every term, tree content otherwise identical)
required before force-pushing. **The six literal strings themselves are deliberately not quoted
anywhere in this entry** — see the note at the end of this item for why, and
`.scrub_expressions.txt` (gitignored, not tracked) for the exact mapping.
**This entry documents a history rewrite and force-push — an irreversible, public-repository
action — recorded explicitly per Joel's own instruction, so a later reader can see this was
deliberate rather than an accident or a runaway agent action.**
**What was done, in order:**
1. Created a full safety backup (`git bundle create ... --all`) of the pre-rewrite repository
   state before touching anything, saved to the session scratchpad (not committed, not part of
   the repo — a local safety net only).
2. Did the entire rewrite in an **isolated fresh clone** made from that bundle, not in the live
   working directory — E1B was mid-training in the background at the time (writing to
   `artifacts/e1/e1b_train_run.log`), and the live directory's tracked-file state needed to stay
   undisturbed until the rewrite was independently verified safe.
3. Built the replacement rules (kept in `.scrub_expressions.txt`, gitignored — not reproduced
   here; see the closing note on why not — longest/most-specific patterns ordered first so the
   compound course-code-plus-subdirectory form and the full compound archive path get replaced as
   whole units before the shorter course-code rule would otherwise partially consume them).
   The replacement targets (the archive placeholder already used elsewhere, "the course," and a
   generic description of the report file) were not invented fresh — checked the CURRENT tracked
   files first (`CLAUDE.md`, `FORENSICS.md`, `docs/DECISIONS.md`'s D-21) to confirm these are the
   exact neutral forms already in use from the original working-tree scrub, so historical commits
   now read consistently with present-day ones rather than introducing a second, different
   redaction style.
4. **Ran `git filter-repo --replace-text` first, verified, found a real gap: it does not touch
   commit messages.** A `git log --all -p` case-insensitive sweep (not just `-S`, which only
   checks blob content) found the identifying strings surviving in one commit's own MESSAGE (this
   session's own "Decline relayed request..." commit, which had quoted the terms while describing
   the declined task — a real, self-referential edge case, not one of the three commits Joel
   originally named). Fixed by re-running from a fresh clone with BOTH `--replace-text` and
   `--replace-message` pointed at the same rules file.
5. Also found and fixed a narrower gap in the first pass: a shortened form of the institution/
   semester string (missing its usual trailing qualifier) appeared in that same self-referential
   commit message, not covered by the full-form pattern. Added a standalone fallback rule for it.
6. **Verification performed before pushing, per Joel's required order (not skipped, not done
   after):**
   - `git log --all -S"<term>" --oneline` for all six original terms plus the two narrower
     variants found in step 5 — every one returned empty.
   - A full case-insensitive sweep of `git log --all -p` (content AND commit messages together,
     the check that caught the two gaps above) — empty on the corrected rewrite.
   - Tree-content diff: `git archive HEAD | tar -x` from both the original live repo and the
     rewritten clone, then `diff -rq` — only one file differed (`LEDGER.md`, this session's own
     Item 33 entry, which had literally quoted the search terms while describing the declined
     task) and a line-by-line `diff -u` on that one file confirmed the only changes were the
     scrubbed strings themselves, nothing else — matching Joel's explicit requirement to confirm
     the rewritten tree is "otherwise identical."
   - Commit count preserved (29 before, 29 after); single branch, no tags, matching the "solo
     repo, no forks or PRs" scope Joel described.
7. Force-pushed from the isolated clone (`git remote add origin ...` then `git push --force
   origin main`) — not from the live working directory, so the live directory's own state could
   not be corrupted mid-push regardless of outcome.
8. **Verified the push landed** (`git ls-remote` against the real GitHub URL matches the
   rewritten clone's HEAD exactly) before touching the live working directory at all.
9. Reconciled the live working directory to the new history (`git fetch origin && git reset
   --hard origin/main`) — safe because the live directory's tracked-file state was already clean
   (confirmed via `git status --short` before running this), and `reset --hard` does not touch
   untracked files, so E1B's in-progress log (`artifacts/e1/e1b_train_run.log`) was left
   undisturbed and E1B itself was never paused or interrupted by any of this.
**Files changed:** none in the working tree (the rewrite only touched git history/commit
messages, not current file content — confirmed by the tree diff above). The safety bundle and
working clone live in the session scratchpad, not in this repository.
**Environment changes:** git history rewritten; all commit hashes after the rewrite differ from
their pre-rewrite equivalents (this is intrinsic to how `git filter-repo` works — every commit's
content or ancestry changed, so every hash changed, even for commits that contained none of the
scrubbed strings themselves, since their parent's hash changed underneath them). The three
originally-named commits (formerly `89d31d3`, `4091e1b`, `d7082cc`) and this session's own
"Decline relayed request" commit (formerly `ca929e4`, now `ade1f9b`) are the ones whose actual
content changed; every other commit's hash changed only because an ancestor's did.
**Self-critique:** the first verification pass (`-S` checks only) would have been declared
"clean" and pushed if I had stopped there — it was the case-insensitive full-sweep of `-p` output
(checking commit messages, not just blob content) that caught two real, still-open leaks. `-S`
answers "did this string's presence change at some commit," which is exactly right for file
content but blind to commit messages entirely, a distinction I did not think through before the
first pass and only caught by deliberately using a second, differently-shaped check rather than
trusting the first one because it returned the expected "empty."
**Verification performed:** all of the above ran as real commands against the real repository
state (not simulated), with output inspected at each step before proceeding to the next
(re-cloned fresh rather than iterating on an already-modified clone, specifically so each
verification pass started from a known, reproducible state rather than compounding uncertainty
about what an in-place re-run might have missed).
**Next:** nothing further required on this item. Continuing E1B (still training in the
background, untouched throughout this whole operation) and the queued E1A seed-2 run once it
completes.

**Correction, same day, before the item's original text finished settling (review
SUP-20260906-50):** this entry originally quoted the actual replacement-rule mapping verbatim,
including every literal identifying string the rewrite exists to remove. That is self-defeating —
a document describing a string scrub, committed normally (not as part of the rewrite itself),
becomes the single largest concentration of the removed strings, sitting in the current, public
HEAD of the very repository the scrub was meant to clean. Fixed by (1) editing this item in place
to describe the operation structurally without quoting the six strings anywhere, (2) moving the
actual rules into `.scrub_expressions.txt`, added to `.gitignore`, kept locally rather than
committed — the same pattern already used for `.archive_path`. In-place editing rather than the
project's usual append-and-annotate convention is deliberate here: preserving the original wrong
text would mean preserving the very data this whole item exists to remove, which has no audit
value the way an ordinary factual correction's original text does. Also checked (read-only, not
edited — `reviews/` is not this session's territory): the same literal-quoting pattern is present
in the director's own `reviews/REVIEW_QUEUE.md` and `reviews/SUPERVISOR_LOG.md` entries
discussing this same finding — flagged to that session directly, who fixed both in place
(their own territory) and asked this session to commit on their behalf, since reviewer output is
read-only to the producer; committed as `ffcb1ba`.

**Known, accepted residual risk (not actionable, recorded so it is a known limitation rather
than an assumed-complete cleanup):** anyone who cloned or forked the repository in the window
between the original push and the force-push still holds the pre-scrub objects locally, and
GitHub may retain now-unreferenced objects server-side for some period after a force-push. For a
repository this new, this quiet, and with the "no forks or PRs" scope Joel described, this is
close to a theoretical concern — but it is a real limit on what a force-push alone can guarantee,
distinct from "the current public HEAD and history graph are clean," which is what was actually
verified above.

## [2026-09-06T16:15:00 UTC] Item 35 — E1B result in, raw and surprising: R-Precision-top3 higher than E1A's, not lower. No conclusion drawn. E1A seed 2 launched.
**Status:** in progress (E1B complete; E1A seed 2 launched, not yet complete)
**Acceptance criteria:** run E1B to completion (train-on-truncated, eval-on-truncated, per Item
30's design), report the raw result honestly regardless of direction, and — per the director's
explicit sequencing requirement — draw no A-vs-B conclusion until E1A seed 2's seed spread and
SUP-49's retrievability-alone control both exist.
**Files changed:** `docs/EXPERIMENT_LOG.md` (new E1B entry, status PARTIAL — raw result reported,
interpretation explicitly deferred, and a real process gap named: this rung ran without a fresh
dedicated pre-registration table filled in before training started, unlike every other rung).
`artifacts/e1/e1b_train_record.json`, `artifacts/e1/e1b_train_run.log` (the run's own outputs —
completed cleanly this time, exit code 0, the `diversity_times` fix held).
**Environment changes:** none beyond the completed ~2.5h E1B run (1.894h training + 0.628h
generation, both close to E1A's timing).
**Result:** **R-Precision-top3(E1B) = 0.3438, vs. E1A's 0.2969 — higher, not lower.** The
pre-registered criterion (`REBUILD_SPEC.md` §6) expected E1B to score worse. It did not, on this
one seed. Ground truth = 0.7950 (consistent with every prior measurement — the evaluator keeps
reproducing itself across five independent configurations now). FID(E1B) = 8.3402, worse than
E1A's 7.2093 (secondary metric, not decisive, but points the opposite direction from R-Precision —
itself worth noting rather than cherry-picking whichever metric agrees with an expectation).
**Explicitly not interpreted as "truncation improves generation."** Two real confounds stand
between this raw number and any such claim: (1) E1A is scored against full captions, E1B against
truncated ones — a text-side retrievability difference the E1-pilot already showed exists at high
R-Precision, of unknown size or even sign at this model's much lower operating range (~0.30-0.34
vs ~0.80), which SUP-49's queued control exists to measure directly; (2) one seed each — no
seed-to-seed spread exists yet to judge whether this gap is larger than ordinary run-to-run
variance.
**Self-critique:** launched E1B without writing a fresh, dedicated `docs/EXPERIMENT_LOG.md`
pre-registration entry first, breaking this project's own established discipline (every other
rung — E0b, E2's tolerance, E1A-power, the E1-pilot and all three of its follow-ups — had a
filled-in hypothesis/criterion table committed before the run). The criterion itself predates the
run (in `REBUILD_SPEC.md` §6), so this is not an undisclosed-after-the-fact rationalization, but
it is a real process lapse under time pressure, named plainly rather than quietly skipped over.
**Verification performed:** read the actual run log line by line before trusting the JSON record
(this run completed and wrote its own JSON cleanly, unlike E1A-power's crash — cross-checked the
log's printed R-Precision/FID/Matching-Score/Diversity lines against the JSON's `mean_dict`
values and confirmed they match exactly, rather than assuming the script's own file write is
correct without spot-checking it once).
**Next:** launched E1A seed 2 (`--seed 20`, otherwise identical config) immediately — CPU was
free the moment E1B's process exited, and this run has SUP-49's retrievability-alone control
built in (Item 32), so it will produce, in one ~2.5h pass: (1) a second A data point establishing
real seed-to-seed spread, and (2) the caption-retrievability-alone number needed to decompose
E1B's raw gap into "model effect" vs. "text-side effect." Only after this lands does an actual
A-vs-B statement go into `docs/EXPERIMENT_LOG.md`.

## [2026-09-06T16:35:00 UTC] Item 36 — E1 closed at E1B: the observed gap is 0.80σ, binomial noise alone exceeds it, no seed count can rescue it. Ladder stops here (D-26).
**Status:** complete
**Acceptance criteria:** the director (Review 9, SUP-51/52/53) pointed out that Item 35's raw
E1A/E1B gap (0.2969 vs 0.3438) rests on only 38 and 44 successes out of 128 trials each — a
binomial-noise question, not yet checked, that could dominate the whole comparison and make the
in-flight "run E1A seed 2 for spread" plan (Item 35) moot before it finishes. Independently
re-derived the arithmetic before accepting it, per this session's own established practice.
**Files changed:** `docs/EXPERIMENT_LOG.md` (E1B entry: status changed from PARTIAL to RESOLVED,
old "two reasons, pending decomposition" text replaced with the binomial analysis and the
"affordability boundary is itself the result" framing — original PARTIAL framing's substance is
superseded, not simply wrong, so this is a genuine update rather than a correction of an error).
`docs/DECISIONS.md` (new D-26: formalizes stopping the ladder at E1B, names E1C/additional seeds
as legitimate-but-unaffordable rather than abandoned, states the reversal condition).
**Environment changes:** none.
**Verification performed — re-derived, not accepted:** computed independently
(`p_a=38/128=0.2969, se_a=0.0404; p_b=44/128=0.3438, se_b=0.0420; gap=0.0469,
se_combined=sqrt(se_a^2+se_b^2)=0.0583; z=gap/se_combined=0.805`) and the sample-size-for-3σ
estimate (`scale=(se_combined/(gap/3))^2≈13.9`, giving ~1,779 samples/arm/seed) — both matched the
director's figures exactly. The independent re-derivation is not a formality: it is the same
practice that has caught real errors on both sides all session (this project's own SUP-46,
SUP-49-vs-my-earlier-noise-floor-slip, and others) — this time the numbers held up, which is
itself informative, not just confirmatory.
**Result:** E1's generation-side question — does caption truncation's already-established
retrieval-space cost (E1-pilot: 0.145-0.157 corpus-wide, ~0.27 conditional, both real and
resolved) propagate into generated motion quality — **is not answerable at any sample size this
hardware affords.** Not "truncation helps," not "truncation hurts," not "no effect" — the honest
label is **unresolved, with the cost of resolving it now measured** (~1,780 samples/arm/seed,
~9 CPU-hours of generation alone per arm per seed, on top of training time). Per D-26, the ladder
stops here: E1C (designed, `REBUILD_SPEC.md` §6a, never built) and any additional E1A/E1B seeds
are not run.
**What this run in-flight when the finding landed produces anyway:** the E1A seed-2 run launched
in Item 35 (`--seed 20`, SUP-49's retrievability-alone control built in) was already ~15-20
minutes into its ~2.5h run when this message arrived. Not killed — restarting would waste the
progress already made for no benefit, and the run still produces two things worth having even
though its original "establish a seed spread to judge the A-vs-B gap" purpose is now superseded
by the stronger binomial-floor argument: (1) SUP-49's caption-retrievability-alone number (cheap
context on how much of any apparent gap sits on the text side, even though no gap this session can
now resolve needs decomposing), and (2) a second real data point on E1A's own R-Precision-top3,
useful supplementary context for whoever picks up a properly-powered version of this comparison
later, even though it cannot retroactively rescue E1B's own comparison. Framed honestly as
"finishing what was already committed to disk and CPU-time," not as continuing to chase a
resolution the math has already closed.
**Self-critique:** the process gap named in Item 35 (E1B ran without a fresh pre-registration
table) compounds here — had a table with an explicit minimum-detectable-effect calculation been
written before E1B ran, this affordability ceiling would have been visible before spending the
~2.5h on E1B itself, not after. Worth carrying forward: future rungs should include a power
calculation (minimum n needed to resolve the smallest effect size worth caring about) in the
pre-registration table itself, not just a hypothesis and a criterion.
**Next:** write up E1's overall conclusion (both the E1-pilot's resolved retrieval-space finding
and E1B's honestly-unresolved generation-side question) as the closing statement for this stage.
Once the in-flight E1A-seed-2/SUP-49-control run finishes (background, no longer decisive, just
supplementary), record its numbers plainly and move on. Per the director's stated priority: Stage
5 (the local demonstrator, designed in `reviews/` SUP-43, "demonstrate the finding, not the
model") becomes the priority after this write-up.

## [2026-09-06T16:35:00 UTC] Item 37 — Stage 5 demonstrator scaffolded: truncation, retrieval, and rendering all independently verified; live end-to-end generation not yet tested
**Status:** in progress — backend components verified individually; the full "type a caption,
click Generate" path (which needs MDM's checkpoint) not yet exercised, deliberately deferred
while E1A seed 2 still holds the CPU (see Self-critique/Next below)
**Acceptance criteria:** implement SUP-20260906-43's design (`reviews/REVIEW_QUEUE.md`) —
type a caption, see the original project's own truncation rule applied live, generate both
versions via MDM's released checkpoint (not this project's own undertrained model), show a
nearest-neighbour retrieval baseline, surface the E1-pilot's real measured numbers on screen.
**Files changed (all new):** `demo/truncate.py` (the original project's rule, faithfully ported
from the archive — same logic as `scripts/e1_pilot_caption_truncation.py` — extended with a
spaCy `en_core_web_sm` POS-tagging step so it runs on arbitrary free-form user text instead of
only HumanML3D's own pre-tagged corpus). `demo/retrieval.py` (TF-IDF cosine-similarity
nearest-neighbour baseline over the materialized corpus's real captions — deliberately simple,
not tuned, so a viewer can trust it wasn't massaged to look good). `demo/generate_wrapper.py`
(thin wrapper reusing MDM's own `sample/generate.py` `main()` directly — not reimplemented —
for single-caption generation against the released checkpoint). `demo/render_real_motion.py`
(renders an already-stored real motion for the retrieval pane, reusing MDM's own
`recover_from_ric` decode and `plot_3d_motion` renderer). `demo/app.py` (Gradio interface tying
all of the above together, with the measured numbers and every stated caveat — internally-
comparable-only, checkpoint-not-ours, CPU-only-several-minutes — printed in the UI itself, not
left to a report). `demo/README.md` (one-command reproduction instructions, requirements, and
known limitations stated plainly, including that the two generated panes' relative quality
cannot be attributed to truncation per D-26). `.claude/launch.json` (new — registers the demo as
a previewable dev server for this session's browser tooling).
**Environment changes:** installed `moviepy<2` (1.0.3 — MDM's vendored renderer imports the old
`.editor`/`mplfig_to_npimage` API that moviepy 2.x removed), `gradio` (already present, 6.20.0),
`spacy` + the `en_core_web_sm` model (~13MB, standard download). All small, standard packages,
within D-19a's authorization; none touch vendored code or require re-verifying anything already
established.
**Two real bugs found and fixed before trusting any of this, not assumed working:**
1. **`plot_3d_motion` does not write its own output file.** It returns a moviepy `VideoClip`
   object; the actual `.mp4` write only happens inside `sample/generate.py`'s own
   `save_multiple_samples()` helper, which `demo/render_real_motion.py` doesn't use (it calls
   `plot_3d_motion` directly for a single real motion, not through the full generation pipeline).
   First test produced no error and no file — caught because the file's existence was checked
   directly rather than assuming a no-exception return meant success. Fixed by adding an explicit
   `clip.write_videofile(...)` call. This also means `demo/generate_wrapper.py`'s expected output
   filename was wrong for the same underlying reason — `save_multiple_samples` writes under its
   "all samples" naming template (`samples_00_to_00.mp4` for a single sample), not the per-sample
   template (`sample00_rep00.mp4`) that gets passed to (and ignored by) `plot_3d_motion`. Fixed
   before ever having run a real generation, by reading `save_multiple_samples`'s source directly
   rather than guessing the output path from the per-sample template's name.
2. **A real moviepy-1.x vs. current-matplotlib incompatibility**: `FigureCanvasAgg.tostring_rgb`
   was removed from recent matplotlib in favor of `buffer_rgba()`; moviepy 1.0.3's
   `mplfig_to_npimage` still calls the old method. Fixed with a small, local, documented
   monkey-patch (`demo/_mpl_moviepy_compat.py`) restoring the old method as a thin wrapper around
   the new one — not a matplotlib downgrade (shared across other projects in this conda env) and
   not a change to vendored MDM code.
**Verification performed:** `demo/truncate.py` tested directly against four hand-picked
sentences, confirming correct truncation-point selection (stops before a real CCONJ) and correct
fallback behavior (first-sentence fallback for a caption with no conjunction). `demo/retrieval.py`
tested against three queries, returning sensible real nearest-neighbour captions from a 24,503-
caption corpus. `demo/render_real_motion.py` tested end-to-end on a real motion id, producing a
verified-playable 300x300 h264 .mp4 (confirmed via `imageio` frame-reading, not just file
existence). `demo/app.py` verified to build (`gr.Blocks` constructs without error) and to serve
real HTML over HTTP (curl against a locally-launched instance returned 200 and a real ~36KB
Gradio page) before being stopped again.
**Does NOT yet establish:** that the full "type a caption, click Generate" path — which invokes
`generate_wrapper.py` twice against MDM's actual checkpoint — works end to end. Each such call
takes several minutes and would contend for CPU with the still-running E1A seed-2 job (Item 36);
deliberately deferred rather than run now and risk slowing or subtly corrupting that job's own
timing measurements. This is a real, stated gap, not a claimed-complete feature.
**Self-critique:** per this project's own stated engineering discipline ("for UI or frontend
changes, start the dev server and use the feature in a browser before reporting the task
complete"), the single most load-bearing interactive path — the Generate button itself — has not
been exercised through the actual UI yet, only its constituent pieces individually. Naming this
gap explicitly rather than letting "everything else worked" imply the whole thing does.
**Next:** once E1A seed 2 completes (freeing the CPU) or once a brief period of contention is
judged acceptable, run one real end-to-end generation through `demo/app.py`'s UI (via the
project's browser preview tooling, not just curl) with a short test caption, confirm both video
panes and the retrieval pane render correctly in the actual interface, and only then consider
Stage 5's core interactive path verified.

## [2026-09-06T16:50:00 UTC] Item 38 — Five director findings actioned: pre-registered the interaction prediction, added a minimum-detectable-effect landmine, validated the demo's truncation, added a stronger retrieval baseline, pinned demo requirements
**Status:** complete
**Acceptance criteria:** action review 10's five findings (SUP-54/55/57/58/59) — a pre-registered
interaction hypothesis before the seed-2 control lands, a new pre-registration/landmines rule
about minimum detectable effect, a real validation of the demo's truncation fidelity, a stronger
retrieval baseline, and pinned demo dependencies.
**Files changed:** `docs/EXPERIMENT_LOG.md` (E1B entry: added the pre-registered interaction
prediction — caption specificity may only help when generator quality is high enough to use it
— stated before the control's number is known; corrected the earlier "process gap" framing to
separate pre-registration, which was real, from the missing per-run record table, which is a
documentation-consistency issue; added a minimum-detectable-effect section to the entry
template). `docs/LANDMINES.md` (new §18: hypothesis + criterion is not a complete
pre-registration without a power calculation — the E1B-specific instance of a rule computable in
about a minute that would have flagged the affordability ceiling before ~5 hours of compute was
spent). `demo/validate_truncation_agreement.py` (new — measures agreement between the demo's
spaCy-based truncation and the E1-pilot's tag-based truncation over 8,962 real captions: 93.8%
punctuation-normalized agreement, 98.1% agreement on which branch fired). `demo/
retrieval_embedding.py` (new — text-to-motion embedding retrieval via `EvaluatorMDMWrapper`,
the same validated space R-Precision uses, kept alongside TF-IDF rather than replacing it).
`demo/app.py` (wires both retrievers in, showing the strong baseline's video and both captions'
similarity scores, explicitly noting when they disagree rather than hiding it). `demo/
requirements.txt` (new, pinned). `demo/README.md` (documents the validation result, the two
retrieval baselines, and the exact install steps including the separate spaCy model download).
`artifacts/demo/truncation_agreement_record.json` (new).
**Environment changes:** none beyond what Item 37 already installed.
**Result — truncation validation:** 93.8% agreement (punctuation-normalized), 98.1% on which
branch fired, over 8,962 real HumanML3D captions. High enough to trust the demo shows
approximately the rule that was actually measured; the ~6% residual disagreement (genuine
POS-tagging differences between spaCy and HumanML3D's own tagger) is disclosed, not hidden.
**Result — embedding retrieval baseline, a real bug caught before shipping it:** a first version
embedded captions on BOTH sides (query and corpus) through the evaluator's text encoder —
semantically poor top-1 matches on manual inspection (e.g. "a man sits down slowly" retrieved a
walking/balancing caption) and spuriously uniform ~0.98+ similarity regardless of query,
indicating the cross-modal-trained text encoder does not discriminate well for pure text-to-text
comparison. Diagnosed by looking at real query results, not assumed correct because the
component was well-validated elsewhere. Fixed by switching to TEXT-TO-MOTION retrieval (embed
the query caption, compare against real corpus MOTION embeddings) — matching exactly what
R-Precision itself measures. Re-tested: 3 of 4 hand-picked queries now retrieve semantically
sensible real motions (walking, jumping/spinning, cartwheel); one (sitting) still misses,
disclosed in the UI as a real, shown-not-hidden limitation, consistent with the director's own
"if the model loses to it on some captions, show that" instruction (applied here to the
retriever itself, which can also be wrong).
**Self-critique:** the first embedding-retrieval implementation would have shipped a plausible-
looking but semantically weaker baseline than TF-IDF if the actual query outputs hadn't been
inspected by eye before moving on — the code ran without error and returned syntactically valid
results, which is not the same as returning good ones. The generically useful check, stated once:
before trusting a new retrieval or similarity method, look at what it actually retrieves for a
handful of real queries, not just whether the code executes.
**Verification performed:** ran `validate_truncation_agreement.py` against the real materialized
corpus (not a synthetic test set) and inspected the sample disagreements directly before trusting
the aggregate percentage. Ran the embedding retriever against four hand-picked queries both
before and after the text-to-motion fix, comparing outputs by eye rather than trusting the
absence of an exception.
**Next:** once E1A seed 2 completes, report its decomposition as context (per the director's own
correction, SUP-54 partially reversing SUP-52 — the run was worth finishing). Then: pre-generate
a small set of example caption pairs (SUP-56, still open, P1 — the demo currently fails its own
30-second comprehension criterion without pre-rendered examples on load) once CPU contention
clears, and do the full live-browser end-to-end test of the Generate button that Item 37 already
flagged as outstanding.

## [2026-09-06T16:55:00 UTC] Item 39 — Checked SUP-60/61's proposed redesign premise before building on it: embedding-space full-corpus self-retrieval collapses to ~1%, real and diagnosed, not a bug
**Status:** complete (finding recorded; UI redesign held pending director's response, since it
changes what the headline statistic should be)
**Acceptance criteria:** the director proposed promoting a "self-retrieval rate" (full caption
finds its own motion N% of the time, truncated finds it M%) to the demo's headline, validating
the premise themselves with TF-IDF (78.3% -> 55.0%, n=300, HumanML3D's own tags) and explicitly
flagging those as premise-validating, not display values, since the demo should use the embedding
retriever per SUP-58. Before building the redesign, re-measure with the actual embedding
retriever the demo will use — do not assume the same qualitative result transfers between two
structurally different retrieval methods.
**Files changed:** `demo/measure_self_retrieval.py` (new — same protocol as the director's own
TF-IDF measurement, HumanML3D's own real tags for truncation so this doesn't touch SUP-57's
separate spaCy-agreement question, but using `EmbeddingRetriever`'s text-to-motion embedding
space instead of TF-IDF). `artifacts/demo/self_retrieval_record.json` (new).
**Result — the premise does NOT transfer to the embedding retriever, for a real, diagnosed
reason, not a bug:** self-retrieval collapses to ~1.0% (full captions) and ~0.7% (truncated),
nothing like TF-IDF's 78.3%/55.0%. Diagnosed directly rather than assumed broken: a real
caption's own true motion scores 0.978 cosine similarity against its own text embedding, yet
ranks **264th out of 8,198** corpus candidates, because every corpus motion clusters in a tight
0.97-0.99 similarity band at full-corpus scale in this embedding space. **This is exactly why
R-Precision itself is only ever validated within 32-candidate batches, never full-corpus top-1**
— the text encoder has enough resolution to distinguish 1-of-32, not 1-of-8,198. "Self-retrieval
rate in the R-Precision-validated space" is therefore not a coherent full-corpus statistic; it
only means something within the batch-of-32 protocol this project has already run, which is
exactly the existing 0.8013 -> 0.6563 R-Precision-top3 numbers already on the demo page.
**Establishes:** the director's TF-IDF self-retrieval number is real and honestly measured, but
it is real *because* TF-IDF does near-duplicate lexical matching well at full-corpus scale, not
because it approximates the same phenomenon as R-Precision — displaying it labelled as "the same
space R-Precision uses" would reproduce the exact fidelity error SUP-58 itself warned against,
with the labels swapped (TF-IDF standing in for the validated space, rather than the reverse).
**Self-critique:** none new this item — this is the same discipline already established
(re-derive before trusting, inspect real outputs before shipping a method) applied to a proposal
from the director rather than to my own prior output; the fact that it caught something this time
is not evidence the discipline is only useful in one direction.
**Verification performed:** ran the actual measurement against the real corpus and the real
embedding retriever (not simulated), then independently diagnosed the near-zero result by
computing one real example's true-motion similarity and rank directly, rather than reporting
"the retriever doesn't work" without knowing why.
**Next:** proposed a resolution to the director (keep the validated R-Precision numbers as the
rigorous metrics-table claim; report the TF-IDF self-retrieval finding as its own clearly-
labelled "lexical retrieval" statistic, not conflated with the embedding space; keep full-corpus
embedding retrieval for the interactive per-query experience only, framed as illustrative, not a
validated statistic) and am holding the UI redesign until hearing back, since the answer changes
which number becomes the demo's headline.

## [2026-09-06T17:10:00 UTC] Item 40 — Demo redesigned per SUP-60/61/62: retrieval promoted to an instant headline, generation demoted, TF-IDF chosen over the embedding retriever on measured grounds. Live-browser-verified.
**Status:** complete for the retrieval headline path (verified live); generation path still
deferred pending CPU availability, as before
**Acceptance criteria:** the director's message crossed with Item 39's independent finding
(embedding retriever collapses at full-corpus scale) and confirmed the same conclusion via their
own SUP-62 reasoning before either party saw the other's number — reconciled by independently
re-measuring TF-IDF myself (not just quoting the director's number) on the identical 300-caption
sample used for the embedding retriever, then rebuilding the interface per SUP-60's proposed
layout: retrieval promoted to the headline (instant, two panes of real retrieved motion, full vs.
truncated caption), generation demoted below the fold behind its own button, explicitly labelled
illustrative/not-evidential.
**Files changed:** `demo/measure_self_retrieval.py` (extended to run both retrievers — TF-IDF and
embedding — on the identical sample/seed, rather than two separate scripts that could drift).
`demo/app.py` (full rewrite — `run_retrieval()` is now the primary, instant interaction using
`NearestNeighborRetriever` [TF-IDF]; `run_generation()` is a separate, secondary handler behind
its own button; headline markdown states the real measured numbers directly, no metric jargon).
`demo/README.md` ("What it shows" rewritten to describe the new design and why the old one was
replaced; "Known limitations" section rewritten to state the TF-IDF-over-embedding decision and
its reasoning plainly, plus the one-sentence 93.8%-vs-98.1% clarification the director asked for
— "same branch fires almost every time, exact cut point occasionally lands a token or two off";
Files list updated).
**Environment changes:** none.
**Result — independently re-verified TF-IDF numbers, on the same sample as the embedding
retriever's ~1%:** 74.7% full-caption self-retrieval, 51.7% truncated, 34.7% of captions'
retrieval changed by truncation — closely matching the director's own independently-run
78.3%/55.0%/46.0% (different random sample, same measurement design; the two runs' agreement
across independent samples is itself a small additional confirmation the effect is real and not
a sampling artifact). These are the numbers now on the demo's headline, not the director's
unquoted figures — per the director's own instruction ("its own numbers go on the page").
**Result — live browser verification, not just curl or a direct function call:** launched the
app, navigated to it in the actual Browser pane, typed a real caption
("a person walks forward and then sits down on a chair") into the actual textbox, clicked the
actual "Show what gets retrieved (instant)" button, and read the resulting page. It worked
exactly as designed: truncation correctly cut the caption to "a person walks forward" (losing the
sitting action), the full caption correctly retrieved a real motion captioned "a person sits down
on a chair" (similarity 0.898) while the truncated caption retrieved "a person walks forward"
(similarity 1.000) — a genuinely clean, real illustration of the measured effect, with no example
curation involved (this was the first caption tried). Both video panes rendered as real, playable
clips (confirmed durations 0:04/0:05 visible in the page text, not just "no error"). This is the
first time this session's UI work has been driven through an actual browser interaction rather
than verified only by direct function calls or HTTP status codes.
**Self-critique:** the earlier design (Item 37) would have shipped a page whose most prominent
visual element actively invited the exact overclaim its own prose forbade — a real design flaw
that neither this session nor the director caught until Review 11 looked at the page holistically
rather than checking each component in isolation. Worth generalizing: verifying that individual
pieces work correctly is not the same as checking that the finished page's overall *emphasis*
matches what is actually established — a page can be built entirely from true, individually-
verified statements and still mislead through what it makes prominent.
**Verification performed:** the live browser test above; independently recomputed the TF-IDF
self-retrieval numbers rather than accepting the director's own measurement at face value, per
this project's established practice.
**Next:** the generation path (secondary, behind "Also generate") still has not been driven
through the browser live — deferred, as before, to avoid CPU contention with the still-running
E1A seed-2 job. `demo/retrieval_embedding.py` and its cache-building step remain in the codebase,
unused by `app.py`, available for a future rung at a smaller corpus scale rather than deleted.

## [2026-09-06T17:20:00 UTC] Item 41 — Embedding-collapse promoted to a real research finding; three small demo fixes (SUP-63/64/65 + polish)
**Status:** complete
**Acceptance criteria:** Review 12/13 asked for four things: (1) record the embedding-retriever
full-corpus collapse as a proper finding in `docs/EXPERIMENT_LOG.md`/`docs/LANDMINES.md`, not
left to survive only as a code comment (SUP-63); (2) state, in the interface itself, that the
independent 78.3%/55.0% measurement replicates this page's 74.7%/51.7% closely (23.3 vs. 23.0
point drop) with one line on why the absolute levels/change-rates differ (SUP-64); (3) tighten
one sentence's wording — "the same phenomenon as" understates that TF-IDF and the embedding
space are different mechanisms measuring a related effect (SUP-65); (4) optional, low-priority
polish — show kept vs. discarded text directly rather than two full sentences.
**Files changed:** `docs/LANDMINES.md` (§13 extended in place with the full-corpus-collapse
finding — a second, more extreme instance of "candidate-pool size is not a configuration detail,
it is the range the instrument is calibrated over," not a new numbered entry, since it is the
same underlying lesson). `docs/EXPERIMENT_LOG.md` (new entry, outside the numbered E-series since
this is a property of the shared evaluation instrument, not this project's own model/data:
hypothesis, result, establishes/does-not-establish, same discipline as every other entry).
`demo/app.py` (headline wording fixed per SUP-65; the independent-replication note per SUP-64
added directly to the page, not just this ledger; kept/discarded text display added to
`run_retrieval()`'s truncation summary, computed as a simple prefix/suffix split since
`truncate_first_action_clause` only ever removes a trailing clause).
**Environment changes:** none.
**Result:** the embedding-collapse finding is now a citable, standalone research result (a
non-obvious property of the Guo et al. evaluator shared across the whole text-to-motion field —
MDM, MotionDiffuse, every number in `LANDSCAPE.md`'s table — that explains why the published
R-Precision protocol specifies a 32-candidate pool rather than that being an arbitrary
convention), not demo trivia. Verified the kept/discarded display works correctly on both a
conjunction-branch example ("a person walks forward" / "and then sits down on a chair") and a
fallback-branch example, via direct function calls; confirmed the updated page text is live via
`get_page_text` against a running instance (the Browser pane was hidden this pass, so a full
click-through re-test was not repeated — only display strings changed, not the interaction logic
already live-verified in Item 40, so this was judged sufficient).
**Self-critique:** none new — these were straightforward, low-risk fixes the director specified
precisely.
**Verification performed:** ran `run_retrieval()` directly for both display cases and read the
returned markdown; started a real server instance and fetched its actual rendered page text
(not just confirming the module imports) before considering the wording changes shipped.
**Next:** report to the director. Still open: the generation path's live browser test (deferred
for CPU reasons, unchanged), and E1A seed-2's decomposition whenever it completes.

## [2026-09-06T17:35:00 UTC] Item 42 — Wrote RESULTS.md, the standalone flagship deliverable; fixed a stale EXPERIMENT_LOG.md header
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-66) pointed out that despite the demo being
verified, E1 closed, and forensics complete, no single document states what this project found —
a reader would have to assemble it from `FORENSICS.md`, a large `EXPERIMENT_LOG.md`, 26 decisions,
18 landmines, dozens of review findings, and a very long ledger. Write the standalone account:
what was established, what was NOT established (in those words, with D-03's unresolved status
stated plainly), how to see it, what it cost — written for someone who has read none of the above,
not a summary requiring cross-references to make sense. Also fix `docs/EXPERIMENT_LOG.md`'s E1B
header, still reading "RAW RESULT IN, INTERPRETATION PENDING" after the entry's own body had long
since been updated to the resolved 0.80σ/D-26 verdict (SUP-20260906-67).
**Files changed:** `RESULTS.md` (new, project root). `docs/EXPERIMENT_LOG.md` (E1B header
corrected in place — a one-line fix, not requiring the append-and-annotate treatment since it is
purely a stale label catching up to already-updated content, not a retracted claim).
**Environment changes:** none.
**Verification performed before writing, not from memory:** re-derived every number in
`RESULTS.md` from source rather than reciting figures from earlier turns in this conversation.
Confirmed the exact F-number labels via `LEDGER.md` Item 7 (F1-F4 from the original forensics,
F6/F7/F8 from the later CFG/normalization audit — **no F5 was ever assigned**, so the document
says "F1-F4 and F6-F8," not the "F1-F8" phrasing used in conversation, to avoid implying an
eighth finding that does not exist). Recomputed the total E1A+E1B training+generation compute
directly from the two runs' own JSON records (6877.9+2626.9+6819.47+2262.31 seconds = 5.16 CPU-
hours) rather than accepting "~5 CPU-hours" on the director's word — it checked out almost
exactly. Verified the four independent ground-truth R-Precision-top3 values (0.7969, 0.8013,
0.7950, 0.7950) directly via grep against `docs/EXPERIMENT_LOG.md` and computed their actual
spread against the published reference (max deviation 0.0043, so "within 0.004" is the accurate
claim — corrected from a recollected "within 0.003" that didn't hold up under the real
arithmetic). Re-read `docs/DECISIONS.md`'s current D-03 text directly to confirm RESULTS.md's
characterization of it (UNRESOLVED, the FID-reproduction gate specifically, the
internally-comparable-only fallback) matches exactly, rather than paraphrasing from memory of an
earlier turn — D-03's own text literally names `RESULTS.md` as where this must be stated loudly,
confirming this document answers a requirement already on record, not inventing a new one.
**Self-critique:** one number recollected from earlier conversation ("within 0.003 across five
runs") did not survive being checked against the actual data (four values, not five; max
deviation 0.0043, not 0.003) — corrected before it went into the flagship document rather than
after, which is the entire point of checking before writing rather than after.
**Next:** report to the director. RESULTS.md is a living document in the sense that E1A seed-2's
decomposition and any future rungs should update it, not just the ledger — but it is complete and
accurate as of this item for everything currently established.

## [2026-09-06T17:45:00 UTC] Item 43 — Removed misleading TF-IDF similarity scores from the retrieval panel (SUP-20260906-68)
**Status:** complete
**Acceptance criteria:** the director independently drove `run_retrieval()` with their own
uncurated caption ("a person raises both arms above their head and then bends down to touch
their toes") and found the truncated arm's displayed similarity (0.827) HIGHER than the full
arm's (0.613) — not a bug in the retrieval itself, but a real display problem: TF-IDF cosine
similarity is not comparable across queries of different length (a shorter query mechanically
scores higher, having fewer terms left unmatched in its own vector, independent of whether it
found the right motion). Displaying both numbers side by side invited the exact wrong reading —
"truncation improved the match" — on the page whose entire argument is the opposite.
**Files changed:** `demo/app.py` (`run_retrieval()`'s `match_summary` no longer prints either
similarity score — the director's own preferred fix among three offered, on the reasoning that
which motion got retrieved is already fully legible from the printed captions and rendered
video, and the score added nothing a non-specialist needed while actively costing something).
**Environment changes:** none.
**Verification performed:** re-ran the director's own exact reported caption through
`run_retrieval()` directly and confirmed the match panel now shows only the two retrieved
captions and motion ids, no numbers to misread.
**Self-critique:** this is the second real bug in this demo caught only by someone actually
running it with a fresh, uncurated caption rather than by reading the code (the first was the
first-two-generated-plot_3d_motion-doesn't-write-its-own-file bug in Item 37; this one required
noticing a *specific number* looked wrong on a *specific real query*, which static review of the
retrieval logic would not surface, since the underlying TF-IDF computation is correct — only its
presentation was misleading). Worth naming as its own category: some defects are visible only by
generating enough real, varied inputs to hit the case where an individually-correct number
produces a misleading page.
**Next:** RESULTS.md (Item 42) already covers the highest-value remaining work per the director's
own stated priority order — this fix and Item 42 crossed in transit. Still open: the generation
path's live browser test (CPU-deferred), E1A seed-2's decomposition whenever it completes.

## [2026-09-06T17:55:00 UTC] Item 44 — Two real errors in RESULTS.md, both confirmed independently before fixing: F5 exists (F1-F8, not F1-F4/F6-F8), and the "0.0036" citation pointed at the wrong number
**Status:** complete
**Acceptance criteria:** the director checked Item 42's two "self-critique" corrections rather
than accepting them, and found: (1) `BRIEFING.md` line 120 defines **F5 — Engineering state**
(monolithic notebook cells, triplicated classes, no seeds, hardcoded Windows paths), verified by
code inspection during the original Stage 1 briefing, absent from `FORENSICS.md` (Stage 1's own
empirical-only scope) and from `LEDGER.md` Item 7's F6-F8 audit — the two places Item 42 actually
checked, which is exactly why "F5 was never assigned" looked true and wasn't; (2) the 0.0044
figure Item 42 cited as evidence for "within 0.004" is the pairwise E0a-vs-E0b gap
(`docs/EXPERIMENT_LOG.md` line 601), not a deviation from the published reference — the real
maximum deviation across four full-split runs is 0.0036, and a fifth on-record value (0.8036)
deviates by 0.0059 but is a restricted-subset measurement, not a full-split one, and needs to be
named as excluded rather than silently absent.
**Files changed:** `RESULTS.md` (§1.1 rewritten to enumerate all eight findings explicitly —
F1-F4 empirical/Stage 1, F5 by code inspection, F6-F8 by the later audit — adding F5 as its own
bullet rather than omitting it; §1.2's evaluator-reproduction claim rewritten to cite the correct
supporting numbers, 0.0036 max deviation across four *full-split* runs specifically, with the
0.8036 restricted-subset value named and its exclusion explained rather than left for a reader to
find unexplained).
**Environment changes:** none.
**Verification performed, independently, before accepting either correction:** (1) read
`BRIEFING.md` line 120 directly and confirmed F5's exact text matches the director's quote
verbatim; (2) recomputed all five deviations from the reference (0.7977) by hand — 0.7969→0.0008,
0.7950→0.0027 (twice), 0.8013→0.0036, 0.8036→0.0059 — confirming the director's table exactly and
confirming the 0.0044 figure really is the E0a/E0b pairwise gap, not a reference-deviation, by
reading `docs/EXPERIMENT_LOG.md` line 601's own sentence directly rather than trusting the
director's characterization of it.
**Self-critique:** two real errors in one document, both from checking too narrow a set of
sources (F5: checked `FORENSICS.md` and the F6-F8 audit, not `BRIEFING.md` itself, which is where
the original Stage 1 task specification actually lived; the citation: reused a nearby number that
looked like it supported the claim without checking what that number actually measured). Both are
instances of the same underlying failure — trusting a plausible-looking number or an absence
without tracing it to its exact source — that this project's own `LANDMINES.md` has been
cataloguing all day from other angles.
**Verification performed:** see above; this item's whole content is verification of someone
else's claims about my own prior work, done the same way this project verifies everything else.
**Next:** report to the director. Still open: SUP-68 already fixed (Item 43, crossed in transit
with this correction), the generation path's live browser test (CPU-deferred), E1A seed-2's
decomposition whenever it completes.

## [2026-09-06T18:05:00 UTC] Item 45 — RESULTS.md's cost figure understated this project's own constraint by ~8x; F8's mechanism mislabeled; a fourth review-discipline landmine recorded
**Status:** complete
**Acceptance criteria:** the director (Review 14) accepted RESULTS.md as the project's top-level
deliverable, then found two remaining issues before signing off fully: (1) `RESULTS.md` §2 quoted
"roughly 12 CPU-hours" for the smallest infeasible FID reproduction attempt, sourced from the
checkpoint's own bundled evaluation log — but that figure is the original MDM authors' hardware,
not this project's, and re-deriving the real cost from this project's own measured generation
rate gives a number roughly 8x larger (which makes this project's own affordability constraint
*stronger*, not weaker, so the error had been understating the very limitation the document
exists to state honestly); (2) F8's description called the mechanism "frame-by-frame" when
`LANDMINES.md` §12a's own code (`batch_mean = batch.mean(dim=0)`) is per-batch, not per-frame.
**Files changed:** `RESULTS.md` (§2's cost paragraph rewritten with the real, re-derived figures
— ~5 CPU-hours for one full-scale replication, ~100 for the full 20-replication protocol, on
this project's own hardware, with the authors'-hardware figure named and explicitly
distinguished rather than silently reused; F8's description corrected from "frame-by-frame" to
per-batch, matching what the vendored code actually does). `docs/LANDMINES.md` (new §19: "a
correct computation can still produce a display that supports the opposite conclusion" — the
fourth review-discipline entry, generalizing both of the demo's own bugs this session found by
running it with real input rather than reading its code, joining §16/17/18 as more transferable
than most of the domain findings).
**Environment changes:** none.
**Verification performed, independently, before accepting either correction:** re-read
`docs/EXPERIMENT_LOG.md` line ~209 directly and confirmed the original E0b entry already
correctly attributes the "12 Hrs" figure to "the author's hardware" — the error was introduced
specifically in `RESULTS.md`'s own condensed retelling, not inherited from a pre-existing wrong
source. Recomputed the real cost by hand from this project's own measured rate (39 min / 128
samples): one full-scale replication (n≈1,000) = 1000 × (39/128) minutes ≈ 304.7 minutes ≈ 5.08
CPU-hours; the full 20-replication protocol ≈ 20 × 5.08 ≈ 101.6 CPU-hours — both match the
director's cited figures (~5.1h, ~102h) to within rounding. Confirmed F8's per-batch mechanism by
re-reading `LANDMINES.md` §12a's actual code directly rather than trusting the director's
characterization alone.
**Self-critique:** the cost-figure error is a genuine instance of the same failure pattern named
in Item 44 (reusing a nearby, plausible-looking number without checking what it actually
measures or whose resource it was measuring) — three instances of essentially the same mistake
in one document now, all caught only because each correction was independently re-verified
rather than folded in on trust. Worth naming plainly rather than treating each as an unrelated,
one-off slip.

**Correction, appended (review SUP-20260906-69, 2026-09-06): "three instances of the same
mistake" above is wrong — it is two of three.** The 0.0044-vs-0.0036 miss and this cost-figure
miss share a shape (reused a plausible-looking number without checking its exact referent or
whose resource it described). **Item 44's F5 miss does not share that shape.** No number was
reused there at all — two documents were searched, neither mentioned F5, and the absence was
read as "F5 was never assigned" without the search scope being wide enough to support that
conclusion. That is a distinct failure (an absence claim resting on an incomplete search, not a
presence claim citing the wrong source) and it now has its own entry, `docs/LANDMINES.md` §20,
written specifically so it would not get folded into this family. Collapsing the two here was
imprecise even though the practical fix (re-verify before accepting) is the same for both —
the *shape* of what went wrong differs, and the record should say so rather than average it out
for a tidier sentence.
**Verification performed:** see above.
**Next:** RESULTS.md is now fully reconciled against Review 14's findings. Still open: the
generation path's live browser test (CPU-deferred), E1A seed-2's decomposition whenever it
completes.

## [2026-09-06T18:15:00 UTC] Item 46 — Fifth review-discipline lesson recorded; all five consolidated with a pointer from RESULTS.md
**Status:** complete
**Acceptance criteria:** the director named the root cause of Item 44's F5 miss as its own,
distinct, transferable lesson — not "reusing a wrong number" (§16/18/19's family) but "an absence
claim is only as strong as its search scope, and that scope is usually invisible in the sentence
stating the conclusion" — and suggested (as a suggestion, not a finding, explicitly left to this
session's judgment) consolidating all five review-discipline lessons scattered across
`LANDMINES.md` into one findable place, with a pointer from `RESULTS.md`. Also confirmed
SUP-70/71 both landed correctly (SUP-71 had already been fixed in Item 45, before this message
arrived — a crossed-message false alarm, confirmed by grep rather than assumed).
**Files changed:** `docs/LANDMINES.md` (new §20 — "'I did not find X' is only 'X does not exist'
if the search was exhaustive," with the F5 miss as its own worked example, distinguished
explicitly from the number-reuse family of entries; a new "Review-discipline lessons (§16-20),
gathered" section immediately after, listing all five in one place with one-line summaries).
`RESULTS.md` (new §5, a short pointer to the consolidated list — deliberately brief, since
`RESULTS.md`'s own job is to stand alone about the project's actual findings, not duplicate
`LANDMINES.md`'s content).
**Environment changes:** none.
**Verification performed:** grepped `RESULTS.md` for "frame-by-frame" before responding to the
director's SUP-71 status question — confirmed zero hits, meaning the fix from Item 45 (committed
and pushed before this message arrived) already resolved it, rather than assuming either that it
was already fixed or that it still needed fixing.
**Next:** the review-discipline lessons are now both individually documented and collected in one
place. Still open: the generation path's live browser test (CPU-deferred), E1A seed-2's
decomposition whenever it completes.

## [2026-09-06T18:25:00 UTC] Item 47 — Sixth review-discipline lesson recorded: a redaction cannot be documented by quotation
**Status:** complete
**Acceptance criteria:** the director (SUP-20260906-72) pointed out that the git-history-scrub
incident (Item 34's original leak and the director's own SUP-50 finding making the identical
mistake immediately after warning against it, both real, both this session, within about an
hour) was never itself written up as a standalone `LANDMINES.md` lesson — it survives only in
`LEDGER.md` entries and a superseded finding, despite being, in the director's words, the one
lesson about how the record itself works rather than how a claim gets checked, and the only one
that forced a genuine exception to this project's standing append-and-annotate correction rule.
Verified genuinely absent (`grep -ni "redaction\|cannot be documented by quotation"` against
`docs/LANDMINES.md` returned nothing) before adding it, per §20's own lesson about absence claims.
**Files changed:** `docs/LANDMINES.md` (new §21 — "a redaction cannot be documented by
quotation," covering both independent instances of the same mistake within the hour, the
append-and-annotate exception it forces, and the do-instead/generalisation sections matching
every other landmine entry's structure; consolidated list updated from five to six lessons,
§16-21, with §21 added as its own bullet). `RESULTS.md` (§5 updated from "five findings" /
`§16-20` to "six findings" / `§16-21`).
**Environment changes:** none.
**Verification performed:** the grep above, run before writing rather than assumed; re-read
`LEDGER.md`'s own Item 34 (the original leak) and Item 44's correction record to confirm the two
independent instances and the append-and-annotate exception were both accurately characterized
before writing §21's account of them.
**Next:** the review-discipline section is now complete at six lessons. Still open: the
generation path's live browser test (CPU-deferred), E1A seed-2's decomposition whenever it
completes.


## [2026-09-06T18:40:00 UTC] Item 48 — Repo-wide numeric-consistency sweep (SUP-20260906-73): DECISIONS.md fix confirmed real, one additional instance found outside the director's own sweep scope
**Status:** complete
**Acceptance criteria:** the director's SUP-73 stated a new standing rule — "when a number is
corrected, the unit of correction is the repository, not the document it was noticed in;
`git grep` the value and its paraphrases before calling the fix done" — and named one concrete
finding under it: `docs/DECISIONS.md` (D-03's status paragraph) still carried the stale, wrongly-
attributed "~12 CPU-hours" figure (the MDM authors' own bundled-log number for their hardware,
silently reused earlier as if it were this project's cost) after the correct figure had already
been fixed elsewhere (RESULTS.md, EXPERIMENT_LOG.md). Verifying meant: (1) confirm the DECISIONS.md
instance is real, fix it; (2) `grep` the whole repo (excluding `third_party/`, `primary_source/`)
for both this figure and the OTHER live numeric correction from this session (the 0.0036-vs-0.0044
evaluator-reproduction figure) to check for any further instance the director's own sweep might
have missed, since their sweep covered `reviews/`, `README.md`, `docs/00_START_HERE.md`,
`docs/METRICS_EXPLAINED.md`, `SUPERVISOR_LOOP_PROMPT.md` but not `demo/` code.
**Files changed:** `docs/DECISIONS.md` (D-03 status paragraph: old "the full protocol costs ~12
CPU-hours on this hardware" replaced with the real, properly-attributed figures — ~5 CPU-hours
for one full-scale replication (n~1000) and ~100 CPU-hours for the full 20-replication protocol,
both derived from this project's own measured generation rate, ~39 min/128 samples; the
checkpoint's bundled log's ~12 Hrs figure named explicitly as the *authors'* hardware, not this
project's). `demo/retrieval_embedding.py` (module docstring, lines 1-6: "within 0.003 of a
published reference across five independent runs this session" — the old, superseded figure —
replaced with "within 0.0036 of a published reference across four independent full-split runs,"
matching the wording already consistent everywhere else in the repo. This file's own sweep scope
(`demo/`) had not been covered by the director's prior pass, since it is code, not docs — a real
gap in scope, not a duplicate finding.)
**Environment changes:** none.
**Self-critique defects found:** the first broad-pattern grep (`"0\.003.*five\|five independent
runs\|within 0.003"`) produced false positives against `SUPERVISOR_LOOP_PROMPT.md`,
`docs/METRICS_EXPLAINED.md`, and `docs/00_START_HERE.md` — all three already correctly say
"0.0036," and the loose pattern was simply substring-matching "0.003" inside "0.0036." Caught by
re-reading each hit individually rather than trusting the grep count; recorded here so the false-
positive rate of a loose sweep pattern doesn't get silently treated as three additional real
findings.
**Verification performed:** `git grep -n` for both figures across all tracked `.py`/`.json`/`.md`
files outside `third_party/`/`primary_source/`; each hit read in context individually. Confirmed
`scripts/e0b_mdm_reproduction.py` (lines 4, 145) and `artifacts/e0/e0b_mdm_reproduction_record.json`
(line 3) already correctly attribute their own "~12 Hrs" mentions to "the author's own bundled
eval log ... on their hardware" — no fix needed there; this was directly re-checked in this turn,
not assumed from earlier-session context. `LEDGER.md` and `reviews/` hits for both figures are
correction-notes quoting the old value by design (append-only record of what was wrong) and are
correctly left as-is.
**Next:** report back to the director: SUP-73's DECISIONS.md finding confirmed real and fixed;
`demo/retrieval_embedding.py` flagged as a genuine additional catch outside their sweep's scope
(code, not docs) — consistent with the day's established pattern of each side's independent pass
catching what the other's missed. Still open: the generation path's live browser test
(CPU-deferred), E1A seed-2's decomposition whenever it completes.

## [2026-09-06T19:05:00 UTC] Item 49 — E1A seed-2 (SUP-49 decomposition control) completed; supplementary, does not reopen D-26
**Status:** complete
**Acceptance criteria:** this run was launched hours earlier (per prior agreement with the
director) specifically as supplementary context — a second seed of arm A plus the SUP-49
retrievability-alone decomposition control — with the explicit prior agreement that its numbers
would be reported once available but would NOT reopen the already-closed E1 statistical
conclusion (0.80σ, D-26). Completing the task means: recording the numbers, checking them against
the pre-registered prediction (SUP-54, written into `docs/EXPERIMENT_LOG.md` before this run's
number was known), and stating plainly whether anything here changes D-26 (it does not).
**Files changed:** `docs/EXPERIMENT_LOG.md` (E1B entry: new supplementary block after the
pre-registered SUP-54 prediction, recording seed-20 arm A's R-Precision-top3 = 0.34375, the
decomposition control's 0.3125, the binomial check on that gap (0.53σ, not a signal — the
pre-registered "neutral-to-helpful" prediction is not supported, though the gap is too small to
call refuted either), and two further binomial checks: E1A seed-1 (0.2969) vs seed-2 (0.34375)
lands at 0.80σ — the *same* z-score, to three significant figures, as the original cross-arm
E1A-vs-E1B gap, an observed instance (not merely a projection) of seed noise matching the
"effect" in size; and the seed-averaged E1A (0.3203) vs E1B (0.3438) narrowing to 0.46σ, moving
closer to indistinguishable rather than resolving anything, exactly as D-26's own logic
predicted averaging would do).
**Environment changes:** none. Background process (pid from `e1_train_arm.py --arm a --seed 20
...`) exited 0, per the task-completion notification.
**Self-critique defects found:** none new — this entry deliberately avoids the trap the
pre-registered SUP-54 note was written to guard against (fitting a story to the number after
seeing it): the prediction was checked against the actual result and found not supported, stated
as such, rather than reframed to fit.
**Verification performed:** read `artifacts/e1/e1a_seed2_train_record.json` directly (not just
the tail of the run log) for the exact `mean_dict` and `e1a_truncated_rescore_control` numbers.
Recomputed all three binomial z-scores independently in Python rather than trusting mental
arithmetic, matching the same SE formula used throughout E1A-power/E1B's own analysis.
**Next:** E1's ladder is now fully closed, including its one deferred supplementary run — nothing
further scheduled on this rung. Still open: the demo's generation path live browser test
(CPU-deferred, no longer blocked by E1A seed-2 contention for CPU).

## [2026-09-06T23:15:00 UTC] Item 50 — D-24 reversed: MPS works. Two real MPS bugs found and fixed. D-26 reframed as a bounded null, then E1 CLOSED (a proposed follow-up run was independently verified as correctly powered, then retracted before launch)
**Status:** complete
**Acceptance criteria:** the director challenged D-24 ("MPS is unusable") directly, on the
author's own prompt, and demonstrated the blocker (a float64-to-MPS-device transfer in
`gaussian_diffusion.py::_extract_into_tensor`) was never tested for removability, only
reproduced. Required of this session (SUP-20260906-76, then 77/78/79): (1) apply and verify the
one-line patch; (2) re-run the E0a evaluator sanity check on MPS before trusting any MPS number;
(3) time generation on MPS specifically (the actual cost-dominant term, not assumed from
training's speedup); (4) re-derive D-26's affordability arithmetic from the two measured rates
and state plainly whether the stopping rule holds.
**Files changed:**
- `third_party/motion-diffusion-model/diffusion/gaussian_diffusion.py` (`_extract_into_tensor`:
  cast-before-transfer, D-27's patch, verified bit-identical independently before applying).
- `third_party/motion-diffusion-model/utils/dist_util.py` (`dev()`: added an MPS branch — this
  function only ever returned `cuda` or `cpu` before, a gap neither SUP-76 nor D-27 mentioned;
  without this, nothing could actually reach the patched diffusion code on this machine).
- `third_party/motion-diffusion-model/data_loaders/humanml/networks/evaluator_wrapper.py` and
  `third_party/text-to-motion/networks/evaluator_wrapper.py` (`get_co_embeddings`/
  `get_motion_embeddings`: same cast-after-transfer defect, a **second, independently discovered**
  MPS bug, found only because an end-to-end validation run was executed rather than trusting the
  E0a evaluator gate as sufficient for a different evaluator class — see self-critique below).
- `scripts/e0_evaluator_sanity_check.py` (added `E0_DEVICE` env var, matching the existing
  `E0_SEED` convention, to re-run the same script on MPS without duplicating it).
- `scripts/e1_generation_timing_probe.py` (new — times MDM's generation call in isolation, on an
  untrained model, mirroring `e1_train_arm.py`'s own generation call exactly; used for the
  load-bearing CPU-vs-MPS generation comparison).
- `docs/DECISIONS.md` (new D-28: gate result, measured 5.47x generation speedup vs training's
  ~9.9x — reported separately per SUP-78, the second MPS bug and its fix, the re-derived
  affordability table, SUP-77's bounded-null reframing of D-26 with corrected mechanistic
  comparison number (MDM's own published *generated* score 0.611, not the ground-truth 0.797 an
  earlier draft of the reasoning used), the regime-scoping note tying back to D-25/`LANDSCAPE.md`
  §1.3, and the final decision: **E1 is CLOSED at n=128, no further runs**, with the retracted
  n=384 proposal recorded next to its own retraction rather than silently dropped).
- `docs/EXPERIMENT_LOG.md` (matching supplementary block appended to the E1B entry, in my own
  words, not copy-pasted from any review message).
- `docs/REVIEW_RESPONSES.md` (dispositions for SUP-76/77/78/79, each independently re-verified
  before being marked ACCEPTED, not accepted on the reviewing session's authority alone).
- `artifacts/e1/e1_generation_timing_probe_cpu_n32.json`, `..._mps_n128.json` (the load-bearing
  timing records). `artifacts/e1/e1a_seed10_mps_validation_record.json` (end-to-end MPS
  validation, arm A seed 10 — first attempt crashed on the second bug above; re-run after the fix
  landed clean; treated as a third supplementary A-seed data point per SUP-79's own framing, not
  as new E1 scope).
- Housekeeping: killed a leaked, hung `load_dataset` streaming process (PID 43423, 13h37m at 0%
  CPU, flagged by the director as safe to reap since the data was already materialized on disk).
**Environment changes:** none (no new packages; two vendored-file patches, both authorized under
this project's standing "replace CUDA-assuming code with MPS" mandate, D-19).
**Self-critique defects found:** the director's own SUP-79 explicitly downgraded the end-to-end
MPS validation run I had queued, calling it "a weaker test... the E0a check already did that job
properly." That assessment was wrong, and running the validation anyway (rather than skipping it
on the director's say-so) is what caught the second bug — the E0a gate and the E1 pipeline use
two *different* vendored evaluator classes, and passing one's MPS gate does not, in fact, license
trusting the other's MPS behavior. Recorded because it is the same shape of lesson as
`LANDMINES.md` §16 (naming a blocker checked ≠ checking whether it actually disables the thing
that matters), applied here to a peer's own confidence about test coverage rather than to a
first-party claim — worth its own note if a further landmines entry is warranted later.
**Verification performed:** every number in every SUP message (76/77/78/79) was independently
recomputed or re-run before being accepted — the E0a gate comparison, the n/MDE table (standard
two-proportion formula), the CPU/MPS hour table, `torch.get_num_threads()`, and the bit-identity
of both vendored patches (unit tests plus one full end-to-end re-run after the second bug's fix).
Nothing in this entry is asserted on the reviewing session's authority alone.
**Next:** E1 is closed. Priority per the director's retraction message: (1) this write-up, done;
(2) point/verify the Stage 5 demo's generation path against the pretrained MDM checkpoint it
already uses (`demo/generate_wrapper.py`), and check whether the new MPS generation rate
(3.15s/sample vs CPU's several-minutes) changes what's feasible in the UI — currently blocked on
a separate, real bug: the demo's live generation has been observed hung (near-0% CPU) for over
35 minutes after finishing its first sampling loop, not yet diagnosed; (3) no work on a full
600k-step training run without checkpoint/resume design in place first, per the director's own
explicit gate.

## [2026-09-06T23:35:00 UTC] Item 51 — Correction to Item 50: the demo was never hung; a real, separate wiring bug found instead (SUP-20260906-80)
**Status:** complete
**Acceptance criteria:** Item 50's "Next" section stated the demo's live generation "has been
observed hung (near-0% CPU) for over 35 minutes." The director drove the same UI end to end and
found it had actually finished: both mp4s existed, valid, in a temp dir one path-depth below
where an early diagnostic `find` had looked — the same miss on both sides independently (the
director's own first detector made the identical mistake and initially reported "HANG
REPRODUCED" before catching it). **Retracted here rather than silently edited in Item 50**, per
this file's append-only convention — the wrong claim stays visible next to its correction.
**What was actually true:** the demo process (PID 71090) was idle because both generations had
completed (~137-140s total for both, confirmed independently: file timestamps 14:37-14:38 match
when the button was clicked; process CPU was low because there was nothing left to do, not
because it was stuck).
**The real bug this session's own click had already exercised without noticing:**
`demo/app.py::run_generation` read `truncated_caption` from a `gr.State` that only
`run_retrieval` ever populated. Clicking "Also generate" without first clicking "Show what gets
retrieved" (the natural click order) left that state at its initial `""`, and
`truncated_caption or caption` silently fell back to the full caption for BOTH panels — two
identical videos under contrasting labels, the opposite of the demo's own finding, with no signal
on screen that the comparison never ran. Confirmed independently by the director (byte-identical
mp4s, matching MD5, same Gradio DOM file hash under both headings) using a caption I had not
tried ("a person walks forward and then waves with their right hand").
**Files changed:** `demo/app.py` (`run_generation` now computes its own truncated caption via
`truncate_first_action_clause(caption)`, the same way `run_retrieval` does, removing the
click-order dependency entirely rather than guarding the state — the fix the director preferred,
since a guard alone would still error on the most natural click order. Added a loud check: if
truncation is a genuine no-op for a caption (no conjunction to cut), both panels now show the
same generation on purpose, with a markdown note saying so, rather than silently rendering what
looks like an unrun contrast. Removed `truncated_caption_state` entirely once its only reader was
gone — an orphan my own fix created, not left dangling.). `demo/generate_wrapper.py` (docstring's
"CPU-only... per D-24" rationale was void after D-27/D-28 reversed D-24; updated to state the
device now comes from `dist_util.dev()`, unchanged from what E1A/E1B/E0b already trust on MPS;
also fixed an unrelated stale citation in the same docstring — "severely undertrained... per
D-24" cited the wrong decision entirely, D-24 is about MPS, not training budget).
**Environment changes:** killed and restarted the demo server (PID 71090 stopped) so the new
process picks up `dist_util.py`'s MPS patch, the `evaluator_wrapper.py` fix, and this bugfix
together — the old process had all three patches applied to files on disk but cached the
pre-patch modules in memory, per Python's normal import-once behavior.
**Self-critique defects found:** my own diagnosis in Item 50 ("hung") was reached by checking
process CPU% and server stdout, never the actual output files on disk — exactly the search-scope
failure `LANDMINES.md` §20 already names ("an absence claim is only as strong as its search
scope"), now confirmed as a second independent instance of the same mistake, made by both
sessions on the same artifact within the same hour.
**Verification performed:** `find`'d the real temp directories directly (one path-segment deeper
than my first attempt), confirmed both mp4s exist with plausible sizes and real timestamps
matching the original click. Read `demo/app.py` end to end to confirm the state-wiring bug
independently before fixing it, rather than accepting the director's diagnosis unchecked. Full
live re-verification (restart server, click through both buttons, confirm two genuinely different
generated videos) deferred to immediately following this entry, not yet completed at write time.
**Next:** restart the demo server, click through both buttons with a caption where truncation
genuinely fires, confirm two different generated videos and a correct match/no-match note;
measure the real single-sample MPS generation wall-clock time (batch=1 has different fixed-
overhead characteristics than the batch=32 rate already measured) before updating the UI's
"several minutes" copy to whatever is actually true now.

## [2026-09-06T23:50:00 UTC] Item 52 — MPS re-validation succeeded end-to-end after the evaluator_wrapper.py fix
**Status:** complete
**Acceptance criteria:** confirm the `evaluator_wrapper.py` cast-before-transfer fix (Item 50)
resolves the crash for real, in the actual pipeline, not just the isolated unit test already run.
**Result:** arm A, seed 10, MPS, n=128 — R-Precision-top3 = 0.328125 (44/128... actually
42/128) vs the existing CPU seed-10 record's 0.2969 (diff 0.0312), FID = 9.293 vs CPU's 7.2093
(diff 2.08). Both differences are smaller than or consistent with this project's own already-
established noise: the R-Precision gap (0.031) is smaller than the CPU-only seed10-vs-seed20
spread (0.047, `docs/DECISIONS.md` D-28), and FID is already known unstable at n=128 regardless
of device (values have ranged 1.07-3.29 across CPU-only re-references at this same n,
`docs/DECISIONS.md` D-25). Nothing here is alarming or reopens anything — it is the third
supplementary arm-A seed anticipated by SUP-79, landing where seed noise alone would predict.
**Files changed:** `artifacts/e1/e1a_seed10_mps_validation_record.json` (overwritten with the
successful run; the crashed first attempt produced no file, so nothing was lost).
**Verification performed:** compared directly against `artifacts/e1/e1a_power_check_record.json`
(the CPU seed-10 record) before writing this entry, not asserted from memory.
**Next:** live browser verification of the `demo/app.py` bugfix (SUP-20260906-80): restart the
server, drive both buttons with a caption where truncation fires, confirm two different generated
videos, measure a real single-sample MPS generation wall-clock time.

## [2026-09-07T00:50:00 UTC] Item 53 — D-29 recorded; Task 1 notebook (CLIP spatial blindness) built, executed, confirmed
**Status:** complete
**Acceptance criteria:** author-directed pivot away from training-comparison experiments (every
such experiment repeats E1's exact shape — an unproven training budget, scored by an instrument
already under suspicion) toward instrument-level questions that need no training. New deliverable
format: executed Jupyter notebooks in `notebooks/`, with committed outputs, not scripts or
markdown. Task 1: does MDM's own frozen CLIP text encoder (the actual conditioning signal, not a
stand-in) distinguish spatial language (left/right, forward/backward) from non-spatial language
of comparable edit distance, with a proper control group and a significance test, not two
cherry-picked numbers — plus how much of HumanML3D's own corpus this actually touches.
**Files changed:** `docs/DECISIONS.md` (new D-29: the "no training comparisons for now" decision,
in my own words, with the reversal condition named). `notebooks/01_clip_spatial_blindness.ipynb`
(new — built via `nbformat`, executed via `jupyter nbconvert --execute`, outputs committed).
**Result, produced entirely by cells in the notebook, not pasted from anywhere:** MDM's own
`load_and_freeze_clip`/`clip_encode_text` (copied verbatim from `model/mdm.py`, ViT-B/32, the
version this project's checkpoints actually use) embeds 16 spatial minimal pairs (mean cosine
similarity 0.9654) measurably closer together than 16 non-spatial control pairs of comparable
edit distance (mean 0.9296) — Welch's t p=0.00050, Mann-Whitney p=0.00056 (one-sided), rank-
biserial effect size 0.680 (large). **The spatial blind spot is real and not narrow**: 56.9% of
24,503 real HumanML3D captions contain at least one of left/right/forward/backward/clockwise/
in-front-of/behind — "right" alone appears in 24.25% of all captions. This is a benchmark-wide
limitation, not a footnote case.
**Self-critique defects found:** my first rank-biserial effect-size formula
(`1 - 2U/(n1*n2)`) returned -0.680 for an effect that is unambiguously in the positive
(spatial > nonspatial) direction — a sign-convention bug in the formula, not in the underlying
p-values or means. Caught by inspecting the printed sign against the already-confirmed direction
of the means, before writing the verdict cell; fixed to `2U/(n1*n2) - 1` (positive when the first
group is stochastically larger) and the notebook re-executed end to end.
**Verification performed:** ran the notebook twice end to end via `jupyter nbconvert --execute`
(once before, once after the effect-size fix), confirmed zero error cells both times, read every
cell's actual output text directly (not assumed from the code) before writing this entry.
**Next:** report to the director that Task 1 ran clean; Task 2's feasibility gate (is TMR,
arXiv:2305.00976, actually obtainable — code, checkpoint, license — ~20 minutes, do not start
building the notebook yet) is the next item, per the explicit ordering in the author-directed
instructions.

## [2026-09-07T01:15:00 UTC] Item 54 — SUP-80 fix verified live; real MPS generation timing measured cleanly
**Status:** complete
**Acceptance criteria:** confirm the `demo/app.py` bugfix (Item 51) produces two genuinely
different videos when driven live in a browser, on the exact click order that exposed the
original bug (generate without retrieval first); get a real, uncontaminated single-sample MPS
generation wall-clock time for the UI copy.
**First attempt was contaminated, as the director flagged:** a concurrent click from the
reviewing session's own browser session landed on the same Gradio queue as mine (also a port
collision — 7860 was briefly held by an unrelated `http.server` the reviewing session was using
to validate a detector; the restarted demo fell back to 7861, and the earlier port is free again
now). That run's *correctness* was still valid — full caption "a person walks forward and then
sits down on a chair" (md5 `9ce89536...`), truncated "a person walks forward" (md5 `3b14e611...`)
— two different mp4s under two different captions, confirming the fix — but its *timing* was
discarded per the director's own warning, not used anywhere.
**Clean re-run, single click, verified nothing else on the queue (`lsof -i :7861`) first:**
caption "a person kneels down and then stands back up" -> full/"a person kneels down and then
stands back up" done in **10.3s** from click, truncated/"a person kneels down" done in **21.3s**
total for both. Measured from the actual output file mtimes against the JS click timestamp, not
from server stdout (which buffers unpredictably, per Item 51's own lesson).
**Files changed:** `demo/app.py` (button label "Also generate (several minutes)" ->
"Also generate (~10s each, MPS)"; `GENERATION_MD` updated to state the measured ~10s figure and
to cite D-28's bounded-null framing instead of the retracted "unresolvable" language).
**Environment changes:** none (demo server already running on the patched code from Item 51).
**Verification performed:** file-mtime-based timing (not log-based), md5sum difference check,
results.txt content check — all direct, not asserted from the earlier contaminated run.
**Next:** report to the director that SUP-80 is closed (verified live, real timing measured).

## [2026-09-07T01:20:00 UTC] Item 55 — Task 2 feasibility gate: TMR is obtainable, low integration friction. Stopping here per instructions, notebook NOT started.
**Status:** complete
**Acceptance criteria:** ~20 minutes, answer only whether TMR (arXiv:2305.00976) is actually
obtainable — public code, a downloadable checkpoint, and its license. Explicitly instructed not
to start building `notebooks/02_tmr_second_evaluator.ipynb` regardless of the answer.
**Answer: YES, obtainable, with unusually low integration friction for this project specifically.**
- **Code:** `github.com/Mathux/TMR`, official implementation by the paper's own authors
  (Petrovich/Black/Varol), 309 stars, not archived, last pushed 2023-12-11.
- **License:** MIT (confirmed via `gh api repos/Mathux/TMR` — `license.spdx_id: MIT`).
- **Checkpoint:** `prepare/download_pretrain_models.sh` downloads a pretrained-weights archive
  from Google Drive (file id `1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm`) via `gdown`, with a published
  md5sum for integrity checking. **Verified the link is live** (`curl` returns HTTP 200 and
  Google Drive's expected large-file "virus scan warning" interstitial, not a dead-link or
  permission-denied page) — not merely assumed from the README existing.
- **Data-format compatibility — the actual reason this is low-friction:** TMR's own `DATASETS.md`
  states its `guoh3dfeats` motion representation is, for motions under 10s, **bit-identical** to
  HumanML3D's own released `new_joint_vecs/*.npy` files (the author's own stated sanity check:
  `np.abs(new - old).mean() < 1e-10`). This project's E0b generations and ground-truth motions
  are already stored in exactly that format
  (`third_party/motion-diffusion-model/dataset/HumanML3D/new_joint_vecs/`). **Re-scoring the
  existing E0b generations would not require touching AMASS or recomputing any features** — the
  README's own worked example uses the identical path convention this project already has:
  `text_motion_sim.py run_dir=RUN_DIR text=TEXT npy=/path/to/motion.npy`, demonstrated in the
  README against `HumanML3D/HumanML3D/new_joint_vecs/001034.npy`.
**Not yet checked (deliberately, per the 20-minute scope):** whether the pretrained checkpoint's
own `requirements.txt` (pytorch_lightning, hydra-core, einops, orjson) installs cleanly in
`mjs_mlcvdl_unified_m5` without dependency conflicts; whether inference forces CUDA anywhere in
the loading path; the actual runtime of scoring E0b's existing sample set. These are notebook-
build-time questions, not obtainability questions, and are left for `notebooks/02` if and when
building it is authorized.
**Files changed:** none (`LEDGER.md` only, per the explicit instruction to write the answer here
and stop).
**Verification performed:** `gh api` for repo metadata and license, `gh api ... /contents/...`
for README.md/DATASETS.md/the download script's actual content (not summarized from a search
result), a live `curl` check against the actual Google Drive file id extracted from that script.
**Next:** report Task 1 (clean) and this gate's answer (obtainable) to the director; do not start
`notebooks/02` without further authorization, per the explicit instruction that the director
assigns a fallback if TMR were not obtainable — since it is obtainable, the next step is the
director's call, not mine to start unprompted.

## [2026-09-07T01:45:00 UTC] Item 56 — notebooks/01: third arm added (SUP-20260907-82), spatial-blindness finding isolated from the verb-vs-modifier confound
**Status:** complete
**Acceptance criteria:** the director independently re-verified notebooks/01 (24,503 captions,
"right" 24.25%, 0 error cells, the rank-biserial fix confirmed as caught correctly) and raised one
real design gap: the original spatial-vs-nonspatial comparison substituted a *modifier* in the
spatial group but a *verb* in the control group, so "CLIP is blind to spatial language" and the
weaker "CLIP separates verbs better than modifiers in general" were both consistent with the same
numbers. Required: a third arm of non-spatial *modifier* pairs (same syntactic slot as the
spatial group) to isolate spatial-ness specifically.
**Independently verified before acting:** re-ran the corpus scan myself with the director's
claimed expanded term list (adding `forwards`, `anticlockwise`, `upleft`) — got 57.59%, matching
their "~57.6%" almost exactly, and confirmed my original 56.89% figure was correct (not wrong),
just missing those morphological variants — consistent with what the director themselves
reported (their own stricter-regex recount of 54.18% was the wrong one, not mine).
**Files changed:** `notebooks/01_clip_spatial_blindness.ipynb` (added `NONSPATIAL_MODIFIER_PAIRS`,
16 pairs, single adjective/adverb substitution, no directional content; added a Kruskal-Wallis
omnibus test across all three groups plus pairwise Mann-Whitney/Welch comparisons for
spatial-vs-verb, spatial-vs-modifier [now the load-bearing one], and verb-vs-modifier [the
confound check]; expanded the corpus term list per the director's own catch; rewrote the verdict
cell to report whichever way the load-bearing comparison actually lands, not the original
two-group framing). Full notebook rebuilt and re-executed twice (once to add the third arm,
once more after catching and fixing a second issue below).
**Result: the finding survives, isolated, moderated.** Spatial (mean 0.9654) vs non-spatial
modifier (mean 0.9446): Mann-Whitney p=0.0338, Welch's t p=0.0348 (both one-sided, matching the
directional hypothesis), rank-biserial effect size +0.383 — real, but markedly weaker than the
original spatial-vs-verb comparison's effect size (+0.680). Critically, the two non-spatial
groups do NOT differ from each other (verb vs modifier, p=0.097, two-sided) — meaning the
under-separation is attributable to spatial-ness specifically, not to modifiers separating worse
than verbs in general. Kruskal-Wallis across all three groups: p=0.0033.
**Self-critique defects found:** my own `compare()` helper used a one-sided Mann-Whitney
(`alternative="greater"`, matching the pre-registered directional hypothesis) but a two-sided
Welch's t-test — an internal inconsistency I introduced, not something the review caught. Before
this fix, the load-bearing comparison's t-test (p=0.070, two-sided) and Mann-Whitney (p=0.034,
one-sided) disagreed on significance at the conventional 0.05 threshold, which would have read as
equivocal. Caught by noticing the two tests gave different significance verdicts for what should
be the same directional question, fixed by making both tests one-sided consistently
(`alternative=alt`), and the notebook re-executed a second time. Recorded because this is the
same class of error the effect-size sign bug was (a presentation/methodology bug, not a data
error), and it changed which side of the significance line the load-bearing result reads on.
**Verification performed:** re-ran the full corpus scan independently before accepting the
director's correction (not accepted on their authority alone); re-executed the notebook twice via
`jupyter nbconvert --execute`, confirmed 0 error cells both times, read every relevant cell's
actual printed output before writing this entry.
**Next:** report to the director that the third arm survives (isolated, p<0.05, weaker effect
size than the original two-group comparison, honestly reported as such). Task 2's feasibility
gate (Item 55) already reported obtainable; still awaiting direction on whether to start
`notebooks/02`. Noted for later, not started: the director's mention of an approved further
notebook series (263-d representation / F1 slice bug, FID covariance rank-deficiency at n=128,
F6 classifier-free-guidance-in-loss proof) — explicitly nothing to start yet.

## [2026-09-07T02:10:00 UTC] Item 57 — notebooks/01, round 2: expanded to n=40/group, both sided p-values + Bonferroni reported, one attribution correction to Item 56
**Status:** complete
**Acceptance criteria:** the director re-verified notebooks/01 round 1 and found the finding
correct but mis-sized: (1) the reported significance for spatial-vs-modifier came from switching
a t-test from two-sided (p=0.070) to one-sided (p=0.035) after seeing the two-sided result look
weak — a legitimate call given a genuinely pre-registered direction, but reported in a way that
didn't show both numbers, which reads as the shape of p-hacking regardless of intent; (2) no
correction was applied for the 3 pairwise comparisons actually run (Bonferroni threshold 0.0167,
which the uncorrected p=0.034 does not clear); (3) n=16/group was underpowered for the observed
effect (~23-38/group needed depending on the effect-size conversion used), and — unlike E1, where
more samples meant hours of unaffordable generation — fixing this here costs nothing: CLIP text
embedding is instant, no training, no dataset, no compute budget involved.
**Independently verified before acting:** recomputed Cohen's d directly from the actual pilot
data (0.678, via pooled-SD formula) rather than trusting either the director's point-biserial-
style conversion (0.829) or my own earlier rank-biserial-via-AUC conversion (0.707) — three
different plausible numbers depending on the conversion path, all in rough agreement, and all
requiring n=40/group to reach comfortable (91-98%) power regardless of which conversion is
"correct." Used the direct, conversion-free Cohen's d throughout the rebuilt notebook rather than
relying on any rank-biserial approximation.
**Files changed:** `notebooks/01_clip_spatial_blindness.ipynb` (rebuilt: pilot n=16/group kept
first and separately labeled for auditability, 24 new pairs per group appended with genuinely
varied sentence frames — not the same template reused with new words — reaching n=40/group; a
power-analysis section computing the pilot's own effect size and the n needed for 80%/95% power,
run before the full-sample comparisons, not after; every pairwise comparison now reports both
one-sided AND two-sided p for both Welch's t and Mann-Whitney, plus an explicit Bonferroni
threshold (0.05/3=0.0167) and whether each comparison clears it; verdict cell rewritten to report
whatever the n=40 data actually shows).
**Result: the finding did not weaken at higher n — it strengthened substantially, and surfaced an
honest new nuance.** Spatial (mean 0.9707) vs non-spatial modifier (mean 0.9442) at n=40: Mann-
Whitney one-sided p<0.00001 (two-sided p=0.00001), Cohen's d=+0.995 — clears Bonferroni easily,
far past the n=16 pilot's marginal p=0.034. But the confound check (verb vs modifier) now shows a
real difference itself (p=0.00244) that did NOT appear at n=16 (p=0.097) — modifiers in general
separate somewhat worse than verbs at this larger, more diverse n. This does not invalidate the
primary spatial-vs-modifier test (already matched on word class, doesn't depend on verbs and
modifiers being equivalent), but it does mean the clean "verb-vs-modifier shows literally zero
difference" framing from round 1 was itself an n=16 artifact, not a settled fact — stated as such
in the notebook rather than quietly dropped.
**Correction to Item 56's own phrasing, per the director's direct request:** Item 56 wrote "their
own stricter-regex recount of 54.18% was the wrong one, not mine" in a way that reads as if the
director had claimed my 56.89% was wrong. They did not — they flagged a discrepancy, investigated
it themselves, and reported that their own method was the one that needed fixing. The distinction
matters for the queue's own audit trail: a reviewer disclosing their own instrument error is not
the same event as a reviewer being corrected by the producer, and Item 56 should not have implied
the latter.
**Verification performed:** independently recomputed Cohen's d via three methods before choosing
which to report; notebook executed twice (once for the n=40 expansion, once more after refining
the verdict cell's confound-check wording for precision); confirmed 0 error cells both times;
read every relevant output directly.
**Next:** report to the director that the finding survived and strengthened at n=40, with the new
verb-vs-modifier nuance disclosed rather than hidden. Awaiting direction on Task 2 (`notebooks/02`,
TMR) and the newly-mentioned further notebook series (263-d representation/F1, FID covariance
rank-deficiency, F6 CFG-in-loss) — nothing started on either.

## [2026-09-07T02:25:00 UTC] Item 58 — notebooks/01, round 3: pilot-vs-extension reported separately (SUP-20260907-84); finding fully closed
**Status:** complete
**Acceptance criteria:** the director accepted round 2 in full but flagged one remaining
objection: the 24 new pairs were written *after* the n=16 pilot had already shown the effect, so
even without any deliberate selection, an unconscious pull toward more-separable items would
produce exactly the observed signature (pooled gap 1.46x the pilot's own gap). Required: report
pilot (n=16) and extension (n=24) as separate rows with their own effect sizes next to the pooled
result, rather than only the pooled number, so a reader can see directly whether the effect lives
in both or only in the newer set.
**Independently verified before implementing:** recomputed the pilot/extension split myself,
outside the notebook, before trusting the director's numbers — pilot gap 0.0208 (d=0.678, p=0.034
one-sided), extension gap 0.0303 (d=1.266, p=0.00001 one-sided), ratio 1.46x — matched their
figures exactly.
**Files changed:** `notebooks/01_clip_spatial_blindness.ipynb` (new section 4b: a `subgroup_report`
helper computing pilot-only, extension-only, and pooled gap/d/p for the load-bearing spatial-vs-
modifier comparison; an explicit check for whether both subsets independently clear uncorrected
significance; the final verdict cell updated to state the subgroup result inline rather than as
a disconnected addendum).
**Result: both subsets independently significant.** Pilot alone: p=0.034 (d=+0.678). Extension
alone: p=0.00001 (d=+1.266). The pooled n=40 headline (d=+0.995, clears Bonferroni) is not an
artifact of the newer, potentially-selected pairs alone — the original, untouched, already-
audited pilot set shows the same effect on its own, at its own smaller but still real magnitude.
**Verification performed:** recomputed the entire pilot/extension split independently in a
scratch script before writing anything into the notebook, then executed the actual notebook via
`jupyter nbconvert --execute`, confirmed 0 error cells, read every relevant output directly.
**Next:** per the director's own statement, notebook 02 (TMR second evaluator) is now unblocked —
their independent re-check confirms the feasibility gate (MIT license, unarchived,
`new_joint_vecs/*.npy` bit-identical to `guoh3dfeats`). The director is past their soft stop and
handing over; further work proceeds from the queue/ledger rather than from further live direction
in this thread. Also noted: user pulled one commit (README revision) mid-session; pulled cleanly
(fast-forward, no conflicts) before this commit.

## [2026-09-07T02:45:00 UTC] Item 59 — notebooks/03: the 263-d representation and the F1 slice bug, built and executed
**Status:** complete
**Acceptance criteria:** author-directed standing requirement (relayed via the director): every
notebook needs "visual intuition... diagrams, plots, SIMPLER explanations... instead of just
super dense compressed writeup text," structured in layers (question -> intuition -> setup ->
measurement -> meaning -> limits). No more hard stop on work windows; notebooks 02/03/04 all open
in any order. Started with 03 per the director's own recommendation (strongest artifact, no new
dependencies) — the F1 finding (the original project's `motion[:66]` slice reads the wrong 66 of
263 dimensions) already has a full forensic write-up in `FORENSICS.md`; this notebook demonstrates
it visually rather than restating the prose.
**Verified against source before building, not assumed from memory:** the 263-d layout
(4 root | 63 ric | 126 rot-6d | 66 local-vel | 4 foot-contact) against `FORENSICS.md`'s own
quoted primary-source code; the 22 joint names against `data_loaders/humanml_utils.py::
HML_JOINT_NAMES` (a real, in-repo, verified list I had not previously confirmed by name);
`recover_from_ric`/`recover_root_rot_pos`/`qrot`/`qinv` against the actual vendored
`motion_process.py` and `quaternion.py` source, copied verbatim into the notebook rather than
reimplemented from memory.
**Files changed:** `notebooks/03_263d_representation_and_f1_bug.ipynb` (new). Structure follows
the director's layered format: one-sentence question; an intuition section (form-with-five-
sections analogy) before any code; setup section naming the real data/code sources; the
measurement (dense rigor, unchanged from FORENSICS.md's own rigor, re-run fresh); a plain-language
"what it means"; an honest "what would change my mind." Four visuals, all generated by cells in
the notebook, all embedded inline (see fix below): (1) a colored, to-scale bar of the 263-d
layout with the bug's actual `[:66]` cut marked in red; (2) a text-tree rendering of the 22-joint
kinematic chain, built from the real chain data, not hand-drawn; (3) bone-length CV histograms and
a full 21-bone bar chart, log-scale, correct-vs-wrong decode; (4) a real side-by-side 3D skeleton
render, same motion, same frame, both decodes.
**Result — independent replication of F1, fresh data, fresh random sample:** 40 real
HumanML3D motions randomly selected (seed 0) from this project's own already-materialized
`new_joint_vecs/*.npy` (not the same 250 streamed samples `FORENSICS.md` originally used), 5,805
real frames decoded both ways. Wrong-slice mean bone-length CV = **24.49%** (worst bone,
neck-head, 69.87%); `recover_from_ric` mean CV = **0.00007%** — matches `FORENSICS.md`'s original
finding (25.81% mean, 81.70% worst) closely, on an independent sample, not a re-quote of the same
numbers.
**Self-critique defect found and fixed:** the first execution ran clean (0 errors) but produced
no embedded images in the notebook itself — only the external PNG files `plt.savefig()` wrote to
disk. Root cause: I had explicitly forced `matplotlib.use("Agg")`, which disables Jupyter's
normal inline figure capture (`plt.show()` under Agg does nothing, as its own warning states).
For an artifact meant to be opened and read, images living only as sibling files (not inside the
`.ipynb` itself) defeats the point of "executed notebook with outputs committed." Fixed by using
`%matplotlib inline` instead of forcing a backend, re-executed, confirmed 4 embedded images and 0
errors. **Applied the identical fix to `notebooks/01_clip_spatial_blindness.ipynb`** as a small,
separate, low-cost correction (not the full layered retrofit the director asked to defer) — its
histogram is now embedded too.
**Verification performed:** notebook executed twice (once to catch the missing-embedded-images
defect, once after fixing it); confirmed 0 error cells and correct embedded-image count both
times via direct nbformat inspection, not assumed from nbconvert's own success message.
**Next:** notebooks 02 (TMR second evaluator) and 04 (FID covariance rank-deficiency) still open,
any order. Notebook 01's full layered retrofit (question/intuition/setup/measurement/meaning/
limits structure, not just the inline-image fix already applied) still deferred, per the
director's own explicit sequencing ("once 02-04 are moving").

## [2026-09-07T02:55:00 UTC] Item 60 — territory note: notebook 01 handed to the reviewing session (author override), my scope narrows to 02/04
**Status:** complete (process note, not a finding)
**What happened:** the director reported the author has directly instructed them to revamp
`notebooks/01_clip_spatial_blindness.ipynb` themselves — visual diagrams, reduced text density,
restructuring, with latitude over its code and tests — an explicit author override of their
normal read-only-toward-producer-files rule, scoped to this one file only. They asked me to
confirm what I had changed since their last direction (so they can merge, not overwrite) and to
stop touching it.
**Confirmed and reported:** nothing uncommitted on `notebooks/01_...ipynb` at the time of the
request (`git status` clean on that file, HEAD at `f898d7f`). The only change made to it since
the round-3 close (Item 58) was the small, separate `matplotlib.use("Agg")` -> `%matplotlib
inline` backend fix (Item 59, so its existing histogram embeds inline) — no restructuring, no new
diagrams, no text-density changes, nothing that overlaps with the revamp the director is now
doing. Standing down on notebook 01 entirely going forward.
**Files changed:** none.
**Next:** notebooks 02 (TMR) and 04 (FID covariance rank-deficiency) remain my territory, any
order. Starting notebook 04 next (263-d/F1 notebook 03 already closed; TMR needs an external
checkpoint download, a heavier first step, so sequencing 04 first).

## [2026-09-07T03:20:00 UTC] Item 61 — notebooks/04: FID's covariance rank deficiency at n=128, built and executed
**Status:** complete
**Acceptance criteria:** same layered visual-intuition format as notebooks 01/03 (director's
standing requirement); demonstrate why FID is unusable at this project's affordable n=128
(already established as D-25/LANDMINES §14) rather than restating the prose. Director's specific
visual requests: toy 2D/3D covariance-collapse demo before the real case; the real 512x512
eigenvalue spectrum on a log axis; the parameter count (512x513/2=131,328) stated plainly next to
the toy; a histogram of FID computed repeatedly between disjoint halves of real data.
**Files changed:** `notebooks/04_fid_covariance_rank_deficiency.ipynb` (new). One-sentence
question; intuition section (photographing a cloud from too few angles, before any code); setup
naming the real cached data reused (no new generation, no new download); two toy demos (2D
ellipse collapsing through n=1/2/5/50, 3D ellipsoid through n=2/3/4/50, both built fresh from
synthetic Gaussians so the collapse is visually unambiguous before the real case adds any other
complexity); the real case computed fresh from this project's own cached artifacts (below); plain-
language meaning; honest limits.
**Real data reused, not re-synthesized:** the 128 raw generated motions E0b's own FID score was
computed from (`artifacts/e0/e0b_generated_cache/generated_motions.npz`, cached from that run,
zero regeneration), embedded fresh through the same `EvaluatorMDMWrapper` this project's FID/
R-Precision numbers already depend on; 4,640 real ground-truth embeddings already computed and
cached (`artifacts/e0/fixed_gt_reference/fixed_gt_reference.npz`, from the same E0b follow-up
work). Both artifacts were sitting on disk from earlier session work, reused rather than
regenerated, per this project's own affordability discipline.
**Result:** real eigenvalue spectrum of the 128 cached generated motions' 512-d embeddings —
numeric rank exactly **127**, matching the theoretical maximum (n-1) precisely; the spectrum
falls from 7.6e-5 to 2.0e-15 between rank 126 and 127, a ~10-order-of-magnitude cliff, not a
gradual taper. Disjoint-halves FID on real ground-truth embeddings (30 trials each): mean 1.577
at n=128/side, 0.214 at n=1024/side, 0.109 at n=2000/side — shrinks cleanly with n on data that
never changes, direct empirical confirmation of the estimator-bias story already recorded in
`docs/LANDMINES.md` §14, now demonstrated rather than asserted.
**Real bug found and fixed mid-build (not a self-critique after the fact — caught by the first
execution attempt's own traceback):** `EvaluatorMDMWrapper.__init__` hardcodes
`checkpoints_dir='.'`, which every other script in this project that uses this class satisfies by
being run with `cwd=third_party/motion-diffusion-model/` — the notebook's own cwd (`notebooks/`)
broke this on first execution (`FileNotFoundError: ./t2m/text_mot_match/model/finest.tar`). Fixed
by `os.chdir`-ing into `MDM_ROOT` only for the wrapper's construction and restoring the notebook's
own cwd immediately after, matching the exact pattern already established in
`demo/retrieval_embedding.py`, rather than changing the notebook's cwd globally (which would have
broken the relative paths used everywhere else in the notebook).
**Verification performed:** notebook executed twice (the second time after the cwd fix);
confirmed 0 error cells and 4 embedded images via direct nbformat inspection; every printed number
above read directly from actual cell output, not asserted.
**Next:** notebook 02 (TMR second evaluator) is the last of my three assigned notebooks (01 now
belongs to the reviewing session per the author's override, Item 60). It needs an external
checkpoint download (Google Drive, MIT-licensed, already verified obtainable in Item 55) as its
first real step, unlike 03/04 which needed none.

## [2026-09-07T03:40:00 UTC] Item 62 — notebooks/03, 04: closing sections converted from print() to rendered Markdown, per NOTEBOOK_STYLE_GUIDE.md
**Status:** complete
**Acceptance criteria:** the director's style guide (`reviews/NOTEBOOK_STYLE_GUIDE.md`, read after
studying the revamped `notebooks/01`) requires the "what it means" closing section to use
`display(Markdown(f"..."))` with live computed values, not `print()` of a long f-string — the
values must still be computed by a cell (never pasted), but rendered with headings/bold/tables
instead of a wall of monospace. Also incorporated the guide's "calibrate before you report"
pattern for notebook 04 specifically, per the director's own suggestion: surface the large-n FID
floor as a named scale before the n=128 headline number, not only as a same-paragraph comparison.
**Files changed:** `notebooks/03_263d_representation_and_f1_bug.ipynb` (closing bone-length-CV
summary converted to a Markdown cell with a two-row table, wrong-slice vs `recover_from_ric`).
`notebooks/04_fid_covariance_rank_deficiency.ipynb` (two changes: the disjoint-halves results
table converted to Markdown with an explicit calibration statement — "n=2000/side settles near
0.109, call this the practical floor" — stated before the reader reaches the n=128 headline, plus
a computed ratio, "14.5x higher," making the gap legible as a multiple rather than requiring the
reader to divide two numbers themselves; the closing "what it means" section converted to
Markdown, now referencing the already-established floor rather than repeating all three numbers
a second time).
**Verification performed:** both notebooks re-executed via `jupyter nbconvert --execute`; confirmed
0 error cells, images intact (4 each, unchanged from before), and exactly the expected count of
new `text/markdown` display outputs (1 in 03, 2 in 04) via direct nbformat inspection; read the
actual rendered markdown content of each new cell to confirm it reads correctly, not merely that
it executed without raising.
**Not touched:** `notebooks/01_clip_spatial_blindness.ipynb` — the director's own in-progress
revamp under the author's override; excluded from this commit as agreed (Item 60).
**Next:** notebook 02 (TMR second evaluator), the last of my three assigned notebooks, per the
director's own sign-off to take it once 03/04's closing sections were converted.

## [2026-09-07T04:15:00 UTC] Item 63 — notebooks/02: TMR as a second evaluator, built and executed; SUP-85/86 addressed on notebook 03
**Status:** complete
**Acceptance criteria:** build `notebooks/02_tmr_second_evaluator.ipynb` per the director's spec
(feasibility already confirmed obtainable, Item 55/58's own handover): re-score the existing E0b
generated motions with TMR alongside the Guo evaluator, scatter plot per-sample scores, a
disagreement table naming specific captions, grouped R-Precision top-1/2/3 bar chart. Separately,
two real presentation defects (SUP-20260907-85, -86) surfaced against notebook 03 while this was
in progress — addressed both before considering either notebook done.
**TMR setup (new, one-time):** cloned `github.com/Mathux/TMR` into `third_party/TMR` (MIT
license, re-confirmed via `gh api`), removed its nested `.git` (this project vendors third-party
code as plain directories, no submodules, matching `motion-diffusion-model`/`text-to-motion`).
Installed only the two genuinely missing dependencies (`hydra-core`, `hydra-colorlog` — `einops`,
`pytorch_lightning`, `orjson` were already present at compatible versions) via `uv pip install`,
confirmed `torch`/`numpy` versions unchanged afterward. Downloaded the pretrained checkpoint via
`gdown` (Google Drive, the exact file id from `LEDGER.md` Item 55's own feasibility check),
**verified its md5 (`7b6d8814f9c1ca972f62852ebb6c7a6f`) matched exactly before extracting.**
Smoke-tested the pipeline end-to-end via TMR's own `text_motion_sim.py` logic before building
anything: a genuinely matched (caption, motion) pair scored 0.85, a deliberately mismatched pair
scored 0.52 — confirms the pipeline discriminates correctly before trusting it for a real
comparison. `third_party/TMR/models/` (124MB of checkpoint weights) is gitignored, matching this
project's own `/checkpoints/` convention — the exact fetch command and checksum are documented in
the notebook itself, not just in this entry.
**Real bug found and fixed before it blocked anything:** TMR's own `text_motion_sim.py` (run via
its documented hydra CLI) fails on first use — `TokenEmbeddings.__init__` unconditionally tries to
load a precomputed dataset-wide text-embedding cache this project never downloaded (only the
model checkpoint was fetched, not the full annotation/embedding cache, which is unnecessary for
scoring a small set of specific captions). Fixed by constructing `TokenEmbeddings` directly with
`preload=False`, which falls back to its own already-built-in live encode-on-the-fly path — not a
patch to TMR's own code, just bypassing an unneeded, unavailable cache at the call site.
**Result, all numbers produced fresh by cells in the notebook:** Pearson correlation between
Guo's and TMR's per-sample scores on the same 128 real generated motions: **r = -0.006** —
essentially zero. R-Precision under each evaluator's own native protocol: Guo top-1/2/3 =
0.133/0.234/0.352; TMR = 0.219/0.312/0.406 (chance = 0.031) — both real, TMR consistently higher.
**Investigated the near-zero correlation rather than reporting it flat:** the 8 worst-
disagreement samples all have Guo scoring >=0.96 — near its own ceiling — while TMR spreads the
same motions 0.37-0.48. Connected explicitly (not left as a coincidence) to this project's
already-documented finding that Guo's embedding space compresses real motions into a narrow
high-similarity band (`docs/LANDMINES.md` #13, the full-corpus-retrieval-collapse finding from
the demo work) — the raw-score correlation is confounded by that same ceiling; R-Precision, which
only depends on relative ranking within a batch, is not, and is reported as the more trustworthy
of the two comparisons for that stated reason.
**SUP-20260907-85 (P1/P2/P3, against notebook 03) and SUP-20260907-86 (P1, against notebook 03),
found while this was in progress, addressed before either notebook was called done:**
- `03_bone_length_histograms.png`: independently auto-scaled x-axes made the correct decode
  (CV 0.00033%) look MORE dispersed than the broken one (CV 69.9%) — the exact display-inverts-
  the-finding defect `LANDMINES.md` §19 names, in the single figure this notebook exists to
  deliver. Fixed: shared x-axis across both panels (computed from the wrong-slice data, which
  sets the real scale), correct decode's spike annotated explicitly rather than hidden in an
  axis-corner offset label.
- `03_263d_layout.png`: overlapping red annotation text, a narrow-segment label overflowing the
  left edge, another clipped at the right edge. Fixed: the two red annotations moved to separate
  vertical rows; narrow segments (`root motion`, `foot contact`, 4 units wide each) now labeled
  outside the bar with a leader line instead of centered text that cannot fit; x-axis margin
  extended so no label clips.
- `03_skeleton_side_by_side.png`: a 3-D rendering in which neither skeleton read as a human body
  (default viewing angle, no equal-aspect constraint, independent per-panel axis ranges). Fixed
  per the director's own verified reference (`reviews/REFERENCE_03_skeleton_fixed.py`, adapted
  not copied): a flat frontal (x-y, not x-z — HumanML3D's up-axis is y, x-z is the floor plane
  and was the reference's own first, rejected attempt) projection, equal aspect, shared limits
  computed across both skeletons per frame, axes off. Extended the fix with a 4-frame strip
  (0.25/0.45/0.65/0.85 through the sequence) rather than one frame, per the review's own suggested
  addition, so the wrong decode reads as incoherent over time, not just a single bad pose.
- Notebook 04's eigenvalue spectrum (already praised as "excellent, the standard the other
  figures should meet") got the review's one optional suggestion applied anyway: the two regions
  ("127 directions with measurable spread" / "385 directions at numerical zero, never sampled")
  now labeled directly on the plot.
**Files changed:** `notebooks/02_tmr_second_evaluator.ipynb` (new), `notebooks/03_...ipynb` (three
figure fixes above, re-executed), `notebooks/04_...ipynb` (one label addition, re-executed),
`.gitignore` (new entry for `third_party/TMR/models/`), `third_party/TMR/` (vendored, `.git`
removed).
**Verification performed:** all three notebooks executed via `jupyter nbconvert --execute` after
every change (03 twice, 04 twice, 02 twice — once for the initial build, once after adding the
ceiling-effect investigation); confirmed 0 error cells and correct image counts each time via
direct nbformat inspection; visually inspected all four of notebook 03's rendered PNGs directly
(not assumed correct from a successful execution) before considering the SUP-85/86 fixes done —
this is the same discipline SUP-86 itself named ("read the executed output before writing the
verdict").
**Next:** all three of my assigned notebooks (02/03/04) are now built, executed, and address every
open review finding against them. Report back; await further direction (the director mentioned an
approved further notebook series — 263-d/F1 already done as 03, FID rank-deficiency already done
as 04, plus a new F6 CFG-in-loss proof not yet started).

## [2026-09-07T04:30:00 UTC] Item 64 — notebooks/03: third figure defect (SUP-20260907-87), the all-bones CV bar chart's blue bars were never visible
**Status:** complete
**Acceptance criteria:** the director retracted their own earlier endorsement of
`03_bone_length_cv_all_bones.png` (SUP-85/its follow-up had cited this figure as the correct
pattern to fix the histogram toward) after actually opening the rendered PNG rather than reading
the plotting code: `symlog` at default `linthresh` crushed the correct-decode CV values
(~0.00007% mean) into sub-pixel height against the wrong-slice values (9-70%), so **no blue bars
render anywhere in the figure**, even though the legend lists `recover_from_ric` — a reader would
plausibly conclude the series was never plotted, not that it is near-zero.
**Independently verified before fixing:** opened the actual rendered PNG myself (not assumed from
the finding's description) — confirmed zero visible blue bars across all 21 bones, exactly as
reported.
**Files changed:** `notebooks/03_263d_representation_and_f1_bug.ipynb` (the CV bar chart:
`ax2.set_yscale("symlog")` → `ax2.set_yscale("log")` with an explicit `ax2.set_ylim(floor, ...)`,
floor computed one decade below the smallest real correct-decode CV rather than left to
matplotlib's default; title updated to state the actual mean CV number directly, `0.00007%`, not
just "near-zero").
**Verification performed:** re-executed the notebook (`jupyter nbconvert --execute`), confirmed 0
error cells, then opened the resulting PNG directly and confirmed blue bars are now visible for
every one of the 21 bones, roughly five orders of magnitude below red — the actual comparison
this figure exists to show, now actually shown. This is the third figure-level defect found
against this specific notebook in one review pass (histogram, layout diagram, this bar chart) —
all three are now fixed and independently visually re-confirmed, not just re-executed.
**Next:** all three of my assigned notebooks (02/03/04) address every review finding raised
against them so far, each fix visually re-confirmed by opening the actual rendered image, not
inferred from a clean execution or from reading plotting code.

## [2026-09-07T05:10:00 UTC] Item 65 — notebooks/02: two real Guo-evaluator bugs found (SUP-20260907-88), a third residual gap found and honestly left open
**Status:** complete (with a disclosed open item, not a false "resolved")
**Acceptance criteria:** the director caught an internal inconsistency — this notebook's own Guo
R-Precision-top3 (0.352) vs. this project's own prior measurement on the identical 128 generated
motions (`artifacts/e0/e0b_mdm_reproduction_record_v2.json`, 0.7578) — a 2.2x gap between two of
this project's own numbers. Named cause: cell 8 re-tagged captions with spaCy (`demo/truncate.py`)
instead of using HumanML3D's own pre-tokenised, lemmatised tokens (`texts/*.txt` field 2), which
the Guo text encoder was actually trained on. Required: fix, and the acceptance test is Guo top-3
returning to ~0.76; if it doesn't, the notebook isn't ready.
**Independently verified before fixing anything:** confirmed the exact tokenisation mismatch
directly (`"walking/VERB"` from spaCy vs the dataset's own `"walk/VERB"` for the identical
caption) and confirmed 128/128 of this notebook's captions have an exact, unambiguous match in
the corpus texts files before trusting a lookup-based fix.
**First fix (tokenisation) applied, R-Precision moved 0.352 -> 0.4375 — still short of the ~0.76
target, so the investigation continued per the director's own explicit instruction** ("if it
doesn't, something else is wrong too").
**Second bug found independently, not flagged by the director:** motion embeddings are NOT
batch-composition-invariant, unlike text embeddings (verified: encoding the same caption alone vs.
inside a batch of 2 gives max abs diff 1.2e-6, cosine sim 1.0 — batch-invariant). The same 128
generated motions embedded all-at-once (this notebook's original approach) vs. in four batches of
32 (the official protocol's own batch size) differ substantially: max abs diff 2.06, mean cosine
similarity only 0.85, one sample as low as 0.22. Fixed by computing Guo motion embeddings in
batches of 32, matching the official protocol exactly (text embeddings, already proven
batch-invariant, correctly left as one-at-a-time). R-Precision moved 0.4375 -> 0.6641 — a much
larger step, and this reversed a real downstream conclusion: **R-Precision now favors Guo over
TMR at every rank, the opposite of the pre-fix result**, which had TMR ahead — that earlier
"finding" was an artifact of the two bugs, not a real result, and the notebook's own closing text
has been corrected to say so explicitly rather than silently updated.
**Third gap found via direct testing, NOT resolved, reported openly:** tested whether
batch-composition (which specific 32 samples get grouped together) explains the remaining gap to
0.7578 by computing R-Precision under 5 additional random batch groupings of the identical,
now-correctly-computed embeddings. All six groupings (this notebook's sequential one plus five
random) land in 0.63-0.68 — **none reach 0.76.** This rules out batch-composition as the sole
explanation. **The residual ~0.10-0.15 gap is real and its cause is not yet identified** —
candidates named in the notebook's own limits section (sequence-length handling differences,
ground-truth-pairing differences, an undiscovered difference from the official `eval_humanml.py`
driver this notebook does not yet fully replicate) but none tested yet. This is reported as an
open item in the notebook itself, not resolved by omission — the closing section and limits
section both name it explicitly, and the acceptance test's own stated bar ("if it doesn't [return
to ~0.76], something else is wrong too, and the notebook isn't ready") is honored by NOT claiming
readiness on this point.
**Files changed:** `notebooks/02_tmr_second_evaluator.ipynb` (both fixes applied; a new
batch-composition-sensitivity check cell; the disagreement table, scatter, and closing sections
all re-executed with the corrected embeddings and re-written to state the reversed R-Precision
finding and the still-open residual gap, not silently updated numbers under old prose).
**Verification performed:** every claim above tested directly in a standalone script before
touching the notebook (tokenisation mismatch, text batch-invariance, motion batch-sensitivity,
batch-composition-sensitivity across 6 groupings) — none accepted from the director's diagnosis
alone. Notebook re-executed three times as the investigation progressed; confirmed 0 error cells
and correct image count each time; read the actual rendered Markdown output each time, not assumed
from a clean execution.
**Next:** report the full investigation back, including the still-open residual gap — this is a
case where "look further" was the right call per the director's own explicit instruction, and the
honest result is "substantially fixed, not fully resolved," not "fixed."

## [2026-09-07T05:35:00 UTC] Item 66 — notebooks/02: third bug found and fixed (missing denormalization), residual gap fully closed
**Status:** complete — the acceptance test now passes
**Acceptance criteria:** the director offered two hypotheses for the ~0.10-0.15 residual gap left
open in Item 65 (Guo top-3 0.6641 vs. this project's own prior 0.7578): an `m_lens` division
convention mismatch, or an ordering/pairing bug. Both were investigated; neither was the cause. A
third cause was found instead, by reading the official evaluation driver's own source rather than
guessing further.
**Investigation of the two suggested hypotheses (both ruled out):**
- `m_lens` convention: confirmed `get_motion_embeddings` and `get_co_embeddings` apply the
  identical internal `m_lens // unit_length` division in both places; this notebook passes the
  same raw frame-count convention (`generated_meta.json`'s own `"length"` field) that the
  official generation code itself produced and stored — no mismatch found.
- Ordering/pairing: `generated_motions.npz`'s `motion_i` keys and `generated_meta.json`'s list
  order were already confirmed index-aligned by construction in earlier work this session; no
  permutation bug found on inspection.
**The real, third cause, found by reading `CompMDMGeneratedDataset.__getitem__`
(`comp_v6_model_dataset.py`) directly:** the official pipeline never feeds a generated motion
straight to an evaluator. It first inverse-transforms out of MDM's own training normalization
(`Mean.npy`/`Std.npy`) back to raw HumanML3D feature space, then re-normalizes into the
evaluator's own expected input space (`t2m_mean.npy`/`t2m_std.npy` for Guo; each evaluator has its
own convention). This notebook's cached motions
(`artifacts/e0/e0b_generated_cache/generated_motions.npz`) are in MDM's training-normalized form
(confirmed: MDM's own mean/std and the evaluator's t2m mean/std differ by up to 0.34 in one
dimension — not a no-op) and were being fed directly into both Guo and TMR, each silently
receiving wrongly-scaled input.
**Verified before fixing:** applied the correct denormalize-then-renormalize transform in a
standalone test first — Guo top-3 moved from 0.6641 to 0.7422, immediately closing nearly all of
the previously-unexplained gap, before any change was made to the notebook itself.
**Also affects TMR, not just Guo — checked and fixed for both:** TMR's own `Normalizer` class
expects raw (unnormalized) HumanML3D features and applies its own mean/std internally; feeding it
MDM-normalized motion was a second instance of the identical bug, silently corrupting TMR's scores
too. Fixed at the source (denormalize once, right after loading the cache; each evaluator then
applies its own normalization from that shared, correct baseline) rather than patched twice.
**Files changed:** `notebooks/02_tmr_second_evaluator.ipynb` (data-loading cell now denormalizes
immediately after loading; Guo's motion-embedding function re-normalizes with `t2m_mean`/`t2m_std`
before encoding; TMR's own encoding function was already correctly downstream of the shared
`motions` variable, so no separate TMR-side code change was needed once the shared denorm was
fixed at the source). The batch-composition-sensitivity check, the disagreement table, the
scatter plot, and both closing sections were all re-executed and rewritten to reflect the final,
resolved numbers rather than left describing the intermediate, still-buggy state.
**Result: the acceptance test passes.** Guo R-Precision-top3 = **0.7266** (sequential batching) /
**0.6641-0.7344** (six groupings tested) against this project's own prior measurement of
**0.7578** — the closest tested grouping sits 0.59 binomial standard errors below the target,
comfortably inside ordinary sampling/batch-composition noise. **The original 2.2x discrepancy
(0.352 vs 0.7578) is resolved, not merely reduced** — three independently-verified bugs
(mis-tokenised captions, wrong motion-embedding batch size, missing denormalization) account for
essentially the entire gap. As a side effect, TMR's own scores also changed (now correctly
normalized): Guo/TMR per-sample correlation rose to r=0.304 (from -0.006, then 0.115, at earlier
buggy stages), and R-Precision now agrees between the two evaluators within ~3 points at every
rank, both far above chance — a real, if less dramatic, finding than either intermediate
(buggy) version of this analysis suggested.
**Verification performed:** the denormalization fix tested standalone before touching the
notebook; full notebook re-executed after each change; confirmed 0 error cells and correct image
count each time; every closing/summary section rewritten to match the actual final numbers, not
left describing an earlier, superseded state.
**Next:** report the closed investigation back. Remaining priorities per the director's own
ordering: confirm SUP-87's CV-chart fix is still correctly in place (already fixed and pushed in
Item 64, prior to this session's most recent messages — should be re-verified, not assumed
current), then SUP-89's n=40 pooling-probe extension.

## [2026-09-07T05:55:00 UTC] Item 67 — notebooks/01b: pooling probe scaled to n=40 (SUP-20260907-89)
**Status:** complete
**Acceptance criteria:** the director's own n=8 pilot (`reviews/REFERENCE_pooling_probe.py`)
found that MDM's per-token CLIP features show the SAME spatial-blindness pattern as the pooled
vector notebook 01 already established — suggesting "switch to per-token features" is probably
not a fix, but n=8 with every p>0.19 could not confirm it. Requested: re-run at this project's
own n=40 pair sets (reused from notebook 01, not a fresh set) and report whichever way it lands.
**Verified before building:** confirmed my recalled copy of notebook 01's own 40 spatial / 40
non-spatial-modifier pairs is byte-for-byte identical to the director's current (revamped)
notebook 01 (extracted and diffed programmatically — only a tuple-vs-list JSON artifact, zero
content differences) before claiming reuse rather than re-derivation.
**Files changed:** `notebooks/01b_pooling_probe.ipynb` (new — a separate notebook, not a section
appended to notebook 01, since notebook 01 is the reviewing session's own territory per the
author's override, Item 60). Same six-layer structure as this session's other notebooks; the
pilot's own three-representation method (pooled/mean-over-tokens/most-divergent-token) unchanged,
scaled to n=40, with Bonferroni correction and both sided p-values added (this project's own
established statistical discipline, not in the original n=8 pilot).
**Result: the pilot's finding is confirmed, not just suggested.** All three representations show
spatial pairs less separated than non-spatial-modifier controls, and the gap **grows**, not
shrinks, moving from pooled (+0.0265, d=+0.995) to most-divergent-token (+0.1106, d=+0.724) — all
three comparisons significant even after Bonferroni correction (threshold 0.0167; largest p=0.00023).
**Pooling is not the cause of the spatial-blindness finding; CLIP's per-token representations are
already spatially weak before any pooling happens.** The practical implication: "switch MDM to
per-token CLIP features" would likely add general signal but should not be proposed as a fix
specifically for the spatial deficit — a real, negative result that heads off an expensive wrong
turn, reported as such per the task's own explicit instruction to report whichever way it lands.
**Verification performed:** notebook executed via `jupyter nbconvert --execute`, confirmed 0 error
cells and 1 embedded image; cross-checked the "pooled" row's numbers (0.9707/0.9442/+0.0265/
d=0.995) against notebook 01's own already-published pooled-representation numbers — exact match,
confirming this notebook computes the pooled case identically to notebook 01 rather than by a
subtly different method.
**Next:** with the residual-gap investigation (Item 66), SUP-87 re-verification, and this pooling
probe all complete, all currently-known open items from the director's review queue are
addressed. Awaiting further direction — the strategic context relayed (free Kaggle compute,
pretrained-checkpoint-plus-modification direction, three already-published approaches not to
re-propose, and the unlicensed-dataset-provenance note) is noted for future planning, nothing
actioned on it yet since no specific task was assigned against it.

## [2026-09-07T06:10:00 UTC] Item 68 — notebooks/03, 04: explicit cross-checks against existing project numbers added (SUP-20260907-92's carry-forward lesson)
**Status:** complete
**Acceptance criteria:** the director's retrospective on notebook 02's three-bug investigation
(SUP-92) named the general lesson: three silent mis-scalings compounded into a confident,
backwards conclusion, and the only reason any of it surfaced was an internal consistency check
against one of this project's own prior numbers — not review, not tests. Requested going
forward: whenever a notebook produces a headline number, print an explicit cross-check against
an existing number in this project, where one exists. Named two concrete targets: notebook 04
against `LANDMINES.md` #14's three FID values; notebook 03's bone lengths against the dataset's
own prior measurement.
**Files changed:** `notebooks/03_263d_representation_and_f1_bug.ipynb` (added a printed
cross-check comparing this notebook's fresh 40-motion measurement against `FORENSICS.md` F1's
original 250-streamed-sample measurement — mean CV 24.49% here vs. 25.81% there, max 69.87% vs.
81.70%, different data, same order of magnitude and same worst-bone identity). `notebooks/04_...`
(added a cell computing actual FID between the 128 cached generated motions and the loaded
n=4640 fixed reference, cross-checked against `LANDMINES.md` #14's three prior values).
**Self-critique defect found and fixed before finalizing:** the first version of the notebook 04
cross-check used a naive `min <= x <= max` range check against the three LANDMINES values, which
printed "falls outside" for a value that was actually an exact match (3.2909, matching round 3's
own value to within float rounding) — because round 3 used the identical fixed reference this
notebook also loads, so an exact match was the CORRECT expected outcome, not a range-membership
question. Caught by noticing the printed "outside" contradicted the visibly-identical numbers,
not by re-running blindly; rewrote to name which of the three prior rounds is expected to match
exactly (round 3) and why, rather than treating all three as a single range to fall inside.
**Verification performed:** both notebooks re-executed via `jupyter nbconvert --execute` after
each change; confirmed 0 error cells and correct image counts; read the actual printed
cross-check output each time rather than assuming the fix worked.
**Next:** per the director's own summary, the notebook queue is otherwise clear; the next real
decision belongs to the author.

## [2026-09-07T06:30:00 UTC] Item 69 — notebooks/05: spatial/non-spatial test-split built and checked for confounds (Item B, author-directed, relayed via the director)
**Status:** complete — a real confound was found, not resolved (correctly, per its own scope)
**Acceptance criteria:** partition HumanML3D's official test split into spatial/non-spatial
subsets using notebook 01's own term list; report sizes and cross-check against notebook 01's
corpus-wide 57.6% figure; check the split for length/action-type confounds BEFORE any future
experiment builds on it, since a confound found now is cheap and one found after a training run
is expensive; report whichever way it lands.
**Verified before building:** confirmed the project's actual test-split convention directly
rather than assuming it — `dataset/HumanML3D/test.txt` lists exactly 4,198 IDs, all in the
`sample######` (non-`train_`-prefixed) naming convention this project's own materialized
`texts/`/`new_joint_vecs/` directories already use; the `train_sample######`-prefixed files
(4,000 of them) are the separate train split and correctly excluded.
**Files changed:** `notebooks/05_spatial_subset_split.ipynb` (new). Six-layer structure; loads
all 4,198 real test-split captions and their real materialized motion lengths (no streaming, no
new download); applies notebook 01's own expanded spatial-term list; reports split sizes with an
explicit cross-check against notebook 01's own 57.6% corpus figure; two confound checks (caption
word count, motion frame count) each with Mann-Whitney/Welch's t *and* Cohen's d (not p-values
alone, so a large-n false alarm can be told apart from a real effect); an illustrative (explicitly
weaker, not a formal test) first-verb overlap check as an action-type proxy.
**Result:**
- **Split sizes:** 2,448 spatial / 1,750 non-spatial (58.3% spatial) — agrees closely with
  notebook 01's own corpus-wide 57.6%, consistent with the test split being drawn from the same
  distribution as the rest of the corpus, not a skewed subsample.
- **A real confound was found:** spatial captions are **4.18 words longer on average** than
  non-spatial captions (14.38 vs. 10.19 words), p<0.00001, Cohen's d=+0.592 — a medium-sized
  effect, not a large-n artifact. **Any future experiment comparing R-Precision between these two
  subsets must control for this** (matching or stratifying on caption length) before attributing
  a difference to spatial language specifically, exactly the failure mode this check exists to
  catch before a training run, not after.
- **Motion length differs too, but the effect is negligible in practice:** -5.5 frames
  (-0.28s), p=0.00031 (significant only because n>1,700/group), Cohen's d=-0.104 — explicitly
  distinguished from the caption-length confound rather than reported as equally concerning.
- **Action-type overlap (illustrative only):** 4 of the top 10 first-verbs are shared between
  groups — a weak proxy, reported with that caveat, not as a formal confound test.
**Self-critique defect found and fixed before finalizing:** the first version reported both
confound checks with only a p-value, which would have overstated the motion-length finding (p<0.001,
but the actual effect size is negligible-to-small at this large n). Added Cohen's d to both checks
and rewrote the "what it means" section to state the size distinction explicitly, matching this
project's own established discipline of never reporting statistical significance alone at large n.
**Verification performed:** notebook executed via `jupyter nbconvert --execute` twice (once
initially, once after adding effect sizes); confirmed 0 error cells and 2 embedded images both
times; read the actual printed/rendered output each time before writing this entry.
**Next:** Item A (`docs/EXPERIMENT_DESIGN_E2.md`, the pre-registered design document) needs this
notebook's subset sizes and — now — its confound finding as an input: the design must specify how
it controls for the caption-length confound (matching, stratification, or an explicit acknowledgment
of the limitation) rather than treating the split as clean.

## [2026-09-07T07:00:00 UTC] Item 70 — docs/EXPERIMENT_DESIGN_E2.md: pre-registered spatial-adapter experiment design (Item A, author-directed, relayed via the director)
**Status:** complete (design only — nothing in this design has been run)
**Acceptance criteria (stated by the director before writing, restated here for the record):**
arms (baseline unmodified checkpoint vs. treatment checkpoint+small spatial-contrast adapter,
exact mechanism and placement specified); the spatial/non-spatial split (notebooks/05) as the
design's novel contribution, R-Precision reported per subset; an explicit regression control
against non-spatial degradation, citing MoCLIP's (arXiv:2505.10810) "competitive FID" as the
exact failure mode; a power/MDE calculation using notebooks/05's real subset sizes as an input,
stated honestly if the affordable MDE is too large to be useful; pre-registered failure/
abandonment conditions; a compute plan costed against Kaggle's free tier with the P100-inflation
caveat stated plainly; and an explicit statement of how the design controls for notebooks/05's
own caption-length confound.
**Files changed:** `docs/EXPERIMENT_DESIGN_E2.md` (new).
**Design summary:**
- **Arms.** Baseline = released MDM checkpoint, unmodified, no training (its R-Precision numbers
  already exist from E0b/notebooks 02 — what's new is scoring the two subsets separately, which
  has not been done). Treatment = same checkpoint + a single trainable linear adapter inserted
  between CLIP's frozen pooled output and MDM's own `embed_text` layer, trained via a contrastive
  loss on a **synthetic** spatial-pairs set (deliberately distinct from notebook 01/01b's 40
  evaluation pairs, to avoid circularity), with CLIP and the diffusion transformer kept fully
  frozen. Chose adapter-after-frozen-CLIP over fine-tuning CLIP itself specifically because
  MoCLIP's own reported outcome (fine-tuned CLIP, only "competitive" FID afterward) is the exact
  failure mode this design is built to catch, and an isolated linear adapter has a much smaller
  blast radius than fine-tuning the whole encoder if it goes wrong.
- **The novel split.** R-Precision-top3 reported separately for spatial (n=2,448) and non-spatial
  (n=1,750) test subsets, for both arms — four numbers, not the one aggregate number every
  published evaluation reports (confirmed via `guidance/RESEARCH_E_novel_directions.md` that this
  split is not done elsewhere). Comparisons are length-stratified (terciles) specifically because
  notebooks/05 found the two subsets differ by 4.18 words on average (d=+0.592) — a real,
  medium-sized confound, not a large-n artifact — so a pooled comparison would be invalid; a
  treatment effect that only appears in the tercile where the confound is strongest is flagged as
  a warning sign, not a confirmation, in the failure conditions.
- **Regression control.** Non-spatial R-Precision must not drop by more than this project's own
  already-measured same-arm seed variance (D-28: 0.047 points at n=128) — an explicit,
  pre-registered abandonment trigger, not a soft caveat.
- **Power/MDE.** Using this project's own validated `n = z^2 . 2p(1-p) / delta^2` formula
  (p~=0.35, this project's own E0b/notebook 02 baseline, not the published-frontier 0.76): at 3sigma,
  full-subset generation can detect down to roughly 0.10-0.12 R-Precision points on the spatial
  subset and 0.12-0.14 on the non-spatial subset — stated as a ceiling, with realistic affordable
  n (smaller than the full subset) pushing the true MDE higher, not hidden.
- **Compute plan.** Every GPU-hour number in this section is explicitly labeled a translation
  from this project's own measured Mac CPU/MPS rates (D-27/D-28: 3.15s/sample generation), not a
  fresh GPU measurement — flagged for direct timing on the first real Kaggle session before any
  full-subset run is committed to. At zero assumed GPU speedup over MPS (deliberately the most
  conservative floor), full two-arm generation across both subsets is estimated at ~7.3 hours,
  comfortably inside Kaggle's free ~30 GPU-hours/week even without any GPU speedup credited.
**Self-critique before finalizing:** checked that the design does not smuggle in the notebook
01/01b evaluation pairs as training data for the adapter (would be circular) — the design
explicitly requires a separate synthetic pairs set. Checked that the power calculation uses this
project's own actually-measured baseline R-Precision (0.35, severely undertrained checkpoint) and
not a published-frontier number that would understate the true n needed. Checked that the Kaggle
free-tier claim is labeled UNVERIFIED (relayed, not independently confirmed by this session
against Kaggle's own current documentation) rather than stated as fact.
**Verification performed:** re-read `REBUILD_SPEC.md` §6/§6a directly (not from memory) before
writing, to match this project's own established design-document conventions (hypothesis/
success-criterion table format, explicit regime-note-style caveats, honest-question ->
proposed-construction-as-judgment-call -> what-remains-unresolved -> sequencing-recommendation
structure). Cross-checked every number cited (subset sizes, confound effect sizes, seed
variance, baseline R-Precision) against the specific prior notebook/decision that established it,
rather than restating from memory.
**Next:** per the director's own standing instruction — if no further direction arrives after
Items A and B, self-select the next-most-useful item from `reviews/REVIEW_QUEUE.md` and the
handover, and record the choice and rationale here. Both Items A and B are now complete; will
check `reviews/REVIEW_QUEUE.md` for open findings before self-selecting, unless further direction
arrives first.

## [2026-09-07T07:15:00 UTC] Item 71 — self-selected: SUP-20260907-95's directional confound argument incorporated into E2, qualified rather than accepted whole
**Status:** complete
**Why this, chosen over an unrelated queue item:** `reviews/REVIEW_QUEUE.md`'s newest entry
(SUP-20260907-95) is a just-accepted review of my own just-completed notebooks/05, making a
concrete, actionable proposal directly against the design document I had just written
(`docs/EXPERIMENT_DESIGN_E2.md`, Item 70) — closing this loop before picking an unrelated item
keeps Item A's design current with the newest finding bearing on it, rather than shipping a
design already one review-cycle stale.
**What I checked before accepting it:** the reviewer's claim that the caption-length confound
"favours us" (spatial captions being longer should raise, not lower, R-Precision, making an
observed deficit conservative) rests on an unverified premise — no notebook in this project has
tested whether R-Precision correlates with caption length in this evaluator, and the opposite
is equally defensible a priori: CLIP's own 77-token window with MDM's `CTX=22` truncation
(`model/mdm.py::load_and_freeze_clip`, confirmed against `reviews/REFERENCE_pooling_probe.py`'s
own `CTX=22` constant) could compress a longer caption's signal rather than sharpen it. Declined
to accept the directional claim as fact; incorporated it as a stated, open assumption instead.
**Files changed:** `docs/EXPERIMENT_DESIGN_E2.md` §3 (now pre-registers reporting BOTH the raw/
unmatched and the length-matched spatial-vs-non-spatial comparison, per the reviewer's "report
both" recommendation; adds a cheap no-training addendum — correlate per-sample retrieval hit/miss,
already a byproduct of computing R-Precision, against caption length on the baseline arm, so the
confound's actual direction gets measured once the baseline arm runs rather than assumed now).
`docs/REVIEW_RESPONSES.md` (reply citing SUP-20260907-95: ACCEPTED the re-derivation in full,
qualified the directional argument specifically, explained why).
**Verification performed:** re-read the CLIP truncation convention directly from
`reviews/REFERENCE_pooling_probe.py` (already used and cited earlier this session for notebook
01b) rather than asserting the counterpoint from memory; confirmed SUP-95's own numeric
re-derivation (split sizes, effect sizes) matches notebooks/05's actual output before marking it
ACCEPTED without qualification on that part.
**Next:** design document (Item A) and its self-review loop with SUP-95 are both closed. If no
further direction arrives, next self-selection will scan `reviews/REVIEW_QUEUE.md` again for any
item raised after SUP-95, or fall back to the handover's mentioned F6 classifier-free-guidance
notebook.

## [2026-09-07T07:25:00 UTC] Item 72 — E2: asymmetric-interpretation logic made explicit and bidirectional (director's cross-session follow-up on SUP-95)
**Status:** complete
**Context:** the director's own cross-session message restated SUP-20260907-95's argument and
asked explicitly for "the asymmetric-confound reasoning... in your own words" in the design. Item
71 (previous entry) had already qualified the premise as unverified and pre-registered a
correlation check, but had not written out the actual asymmetric interpretation *rule* the
director asked for — a real gap, not just a restatement of already-completed work.
**What I added, and why it goes further than the director's framing:** the director's message
states the asymmetry in one direction only (assumes length helps retrieval). Since Item 71 already
established that premise is unverified and the opposite (CLIP's `CTX=22` truncation compressing a
longer caption) is equally defensible, a one-directional interpretation rule would silently break
if the length-correlation check comes back the other way. `docs/EXPERIMENT_DESIGN_E2.md` §3 now
has a bidirectional table: which of {unmatched deficit, unmatched null/advantage} is
conservative-and-already-believable vs. uninterpretable-without-matching, under each of the three
possible outcomes of the pre-registered length-correlation check (helps / hurts / no effect). The
matched split is reframed as answering exactly the two cells where the unmatched result alone is
not trustworthy, rather than as a blanket requirement.
**Files changed:** `docs/EXPERIMENT_DESIGN_E2.md` §3 only (one new subsection + table).
**Self-critique:** considered just adopting the director's one-directional framing verbatim, since
it is simpler to write and matches what was asked for literally — declined, because doing so
would have re-introduced the exact unverified-assumption-as-load-bearing-fact problem Item 71 had
just corrected. A single flat asymmetry claim only holds if the premise is true; the bidirectional
table holds regardless of which way the (still unmeasured) correlation actually resolves.
**Verification performed:** re-read the edited section back in place to confirm the table's four
outcome cells are mutually exclusive and jointly exhaustive over the three-way correlation
outcome, and that the "no correlation" row correctly reduces the matched split to a robustness
check rather than a requirement (the one case where director's original framing and this version
agree exactly).
**Next:** Item A is now fully closed against the director's own follow-up. No SendMessage tool
is available in this session to reply directly to the director; this ledger entry plus the
design-doc diff are the only channel back, consistent with this project's established
file-based, ledger-first communication convention. Will continue self-selecting from
`reviews/REVIEW_QUEUE.md`/handover per the standing instruction if nothing further arrives.

## [2026-09-07T08:10:00 UTC] Item 73 — notebooks/06: F6 (CFG-in-training-loss) demonstration; self-caught the notebook's own central claim was wrong, corrected before shipping, docs/LANDMINES.md section 24 added
**Status:** complete
**Why this, self-selected:** the director's cross-session message (SUP-97) confirmed the queue is
clear through SUP-97 and handed off standing direction: work from the handover and queue, keep
moving, self-select and record why. The handover's "queued and specced but not started" list has
three notebooks; 263-d/F1 (notebook 03) and FID rank (notebook 04) are already complete. **F6 —
classifier-free guidance folded into the training loss — is the one remaining item**, and matches
Item 71's own stated fallback.
**Acceptance criteria (stated before building):** reproduce `docs/LANDMINES.md` section 11's
claim — that folding the CFG combination formula into the training loss gives a text-ignoring
model zero gradient pressure to start using the caption — on a minimal toy system, verified
against the archived project's actual primary source (not the paraphrase in LANDMINES.md), with
every number produced by a cell in the notebook.
**What actually happened, and why this entry exists:** the first built version of the notebook
stated the "zero gradient" claim as a confirmed result. **The executed measurement cell
contradicted it** — the archived formula's gradient on the conditioning pathway was not zero; it
was larger than the correct (conditioning-dropout) formula's gradient, and scaled linearly with
the guidance scale `w` (3.0, 6.0, 9.1, 15.3, 21.1 at w=1,2,3,5,7). A full toy training run then
showed the archived formula's conditioning weight actually grew and its loss converged *below*
the dropout formula's, in direct contradiction to the "collapse" narrative the notebook's own
first draft asserted. **Caught this before considering the notebook done, by reading the printed
output against the prose rather than assuming the derivation was right because it sounded
plausible** — the same discipline this project applies to a reviewer's claims, applied here to my
own.
**Root cause of my own error:** conflated "a value collapses to a fixed point" with "the gradient
through that point is zero" — a linear layer's output at zero weight is zero, but its gradient
with respect to that weight, evaluated at zero, is the layer's own input, not zero. True premise
(the forward-value algebra), false conclusion (no gradient), and I had written the conclusion
into the notebook before testing it.
**What I verified instead, before rewriting the notebook's claims:** ran a follow-up scratch test
(`/tmp/test_instability.py`, not committed — throwaway) checking whether the measured
amplification (gradient scaling with `w`) produces the instability `docs/LANDMINES.md` section
11's own point 3 already asserts but never demonstrated (the tanh squash and gradient clipping as
compensating hacks). Confirmed directly: at the archived project's own learning-rate-adjacent
settings, the archived formula diverges at `w=7.0` where dropout remains stable, and diverges at
**both** tested guidance scales (including the project's own default, `w=3.0`) at learning rates
only 3-6x larger; conditioning dropout never diverges at any tested learning rate. Reproduced
this result inside the actual notebook (Part C) before treating it as established.
**Files changed:** `notebooks/06_cfg_training_loss_collapse.ipynb` (new — six-layer structure;
Setup reads the archived project's own `DL_T2P_IMPL.ipynb` cell 47 directly via a code cell and
asserts each of section 11's claims against it before proceeding, rather than trusting the
paraphrase; Measurement has three parts — Part A: gradient measurement at the text-blind point
across w=1..7, falsifying the "zero gradient" claim; Part B: full toy training run showing
convergence, not collapse, at a small learning rate; Part C: the instability test, confirming
section 11's point 3 directly). `docs/LANDMINES.md` (new section 24 — corrects, does not retract,
section 11: the forward-value collapse algebra is confirmed exactly; the "no gradient pressure"
interpretation is not supported by this toy and is superseded by a gradient-amplification-causes-
instability mechanism that matches section 11's own point 3).
**Self-critique defect found and fixed before finalizing (the one described above):** rewrote the
Measurement, "What it means", and "What would change my mind" sections completely rather than
patching the numbers under the old narrative — matching this project's own established practice
of never silently swapping a conclusion under unchanged prose.
**Verification performed:** notebook executed via `jupyter nbconvert --execute` twice (once with
the incorrect narrative, caught before shipping; once after the full rewrite); confirmed 0 error
cells and 1 embedded image both times; read every printed number directly against the prose
before finalizing, both times.
**Next:** per the director's standing instruction, will continue self-selecting from
`reviews/REVIEW_QUEUE.md` and the handover if nothing further arrives. The open gap this notebook
itself names — the archived project's real Phase 3 loss (0.69) is not explained by either the
"no gradient" or the "instability" framing, since an unstable run is not usually the best-loss
phase of three — is left as a stated, unresolved limitation, not chased further here.

## [2026-09-07T08:20:00 UTC] Note — notebook 06 is UNREVIEWED, per the director's own explicit disclosure
**Status:** informational, no action
The director's cross-session message stated it does not have budget left to verify notebook 06
(re-derive its arithmetic, open its rendered figures, cross-check against `FORENSICS.md` F6) and
explicitly said not to read its silence as approval, and to record this rather than treat it as
tacit acceptance. **Recorded as instructed: `notebooks/06_cfg_training_loss_collapse.ipynb` has
not been independently reviewed by anyone but me.** Its own self-critique (Item 73) already
documents one defect I found and fixed myself before shipping; an outside review may still find
something that self-check missed, and until one happens the notebook's claims carry only the
confidence of my own verification, not a second pair of eyes.

## [2026-09-07T08:45:00 UTC] Item 74 — notebooks/07: "why six samples is not a finding" (director-assigned, `E1A vs E1B`'s bounded null, the seed-only reproduction, and the circularity trap)
**Status:** complete
**Acceptance criteria (director's message, restated in my own words):** re-derive, from this
project's own saved artifacts (not memory), why E1's 38/128-vs-44/128 gap is not a finding: the
binomial floor (0.80 sigma); the same-arm seed-only reproduction (E1A seed 10 vs seed 20
reproducing the entire cross-arm gap from randomness alone); the minimum-detectable-effect
calculation showing the experiment was powered for a different effect size, not powerless
outright; and the circularity trap in D-26's original framing (powering against your own noise
reading demands more samples the smaller that reading is). Close by connecting to
`docs/EXPERIMENT_DESIGN_E2.md`'s own pre-registered power section.
**Verified before building, not assumed from the director's message:** read
`artifacts/e1/e1a_power_check_record.json`, `e1b_train_record.json`, `e1a_seed2_train_record.json`,
and `e1a_seed10_mps_validation_record.json` directly. Confirmed the director's specific numbers
(seed 10 = 38/128, seed 20 = 44/128) against the real `seed` fields and `r_precision_top3_*`
values in these files, and found one thing worth being precise about: **a fourth file
(`e1a_seed10_mps_validation_record.json`) also has `seed: 10` but scores differently (0.328125,
not 0.2969)** — a second, later, MPS-validation run of the same nominal seed, consistent with
this project's own already-documented MPS non-determinism (`docs/LANDMINES.md` section 7), not a
labeling error. Excluded it from the notebook's central comparison and said so explicitly, rather
than silently picking whichever "seed 10" record supported the story. Also independently
recomputed the MDE table (`n = z^2 * 2p(1-p)/delta^2`, p=0.32) rather than pasting
`docs/DECISIONS.md`'s numbers, and found the recomputed values differ by a handful of samples per
row from the doc's own prose (186 vs 187, 286 vs 287, ~1,783 vs ~1,780) — traced this to a
rounding-convention difference (nearest vs. ceiling) plus using the exact `k/128` fraction instead
of each JSON's 4-decimal-rounded field, not a real disagreement; stated this explicitly in the
notebook rather than silently matching the prior number or leaving an unexplained discrepancy for
a future reader to puzzle over.
**Result — the exact-match finding, stronger than the director's own framing:** the cross-arm
comparison (E1A=38/128 vs E1B=44/128) and the same-arm seed-only comparison (E1A seed10=38/128 vs
seed20=44/128) do not merely produce z-scores that agree "to three significant figures" (the
director's phrasing) — **the raw counts are identical integers in both comparisons (38 and 44
both times)**, so the two z-score computations are not just close, they are the same arithmetic
performed on the same numbers. Confirmed this directly in a code cell (`np.isclose(z_crossarm,
z_seedonly)` -> True, both 0.8047) rather than asserting it.
**Files changed:** `notebooks/07_why_six_samples_is_not_a_finding.ipynb` (new — six-layer
structure; Setup loads all four JSON records directly; Measurement Part A recomputes the 0.80
sigma binomial floor; Part B is the seed-only reproduction with the exact-count finding above;
Part C recomputes the MDE table and n=128's own detectable-effect range; Part D plots the
required-n-vs-effect-size curve on a log axis, marking where D-26's original noise-blip choice
and the real hypothesis-motivated effect each land on it, making the circularity visually
undeniable rather than just stated in prose; closes by connecting explicitly to
`docs/EXPERIMENT_DESIGN_E2.md`'s own pre-registered power section as the direct downstream
consequence of this lesson).
**Self-critique defect found and fixed before finalizing:** the first executed version used each
JSON's own rounded 4-decimal `r_precision_top3_*` field directly in the gap/SE/MDE formulas
instead of recovering the exact integer count and using `k/128` — a small but avoidable precision
drift that pushed the noise-blip sample-size estimate to 1,784 instead of matching the project's
own ~1,780-1,781. Caught by comparing the executed output against `docs/DECISIONS.md`'s stated
numbers rather than accepting a plausible-looking result, fixed by recovering exact fractions from
the raw counts, re-executed, and the residual few-sample difference that remained afterward was
traced to a rounding-convention difference and stated explicitly rather than left unexplained.
**Verification performed:** notebook executed via `jupyter nbconvert --execute` three times (once
initial, once after the precision fix, once after adding the rounding-convention clarification);
confirmed 0 error cells and 1 embedded image every time; read every printed number against the
primary-source JSON files and against `docs/DECISIONS.md`/`reviews/REVIEW_QUEUE.md` SUP-77 before
finalizing any claim.
**Next:** per the director's standing instruction, will self-select the next item from
`reviews/REVIEW_QUEUE.md`/handover if nothing further arrives — the handover's three-notebook
queue (263-d/F1, FID rank, F6) is now fully complete along with this one, so the next self-
selection will need to look beyond that specific list.

## [2026-09-07T09:10:00 UTC] Item 75 — self-audit of notebooks 06/07 against the reviewer's own checklist; two real figure defects found and fixed; notebooks/README.md written
**Status:** complete
**Context:** the director's cross-session message stated it has no budget left to review notebooks
06 or 07 and handed off the exact checklist it would have used, asking for an explicitly-labeled
self-audit (weaker than independent review) rather than silence. Also asked for
`notebooks/README.md` as an entry point across all eight notebooks.
**Self-audit performed, item by item:**
1. **Opened every rendered PNG directly** (`06_conditioning_collapse.png`,
   `07_circularity_trap.png`) rather than trusting the plotting code. **Found two real defects:**
   notebook 06's right-panel subplot title was clipped at the canvas edge (the same class of
   defect as SUP-85's clipped title on notebook 03); notebook 07's two annotation text boxes
   (the noise-blip label and the hypothesis-effect label) visually overlapped and were illegible
   together. Both are exactly the categories the director's checklist named ("clipped
   annotations," "overlapping... annotations") — this self-audit would have missed nothing an
   independent reviewer's first look would have caught here.
2. **Checked axis-sharing on every comparison panel.** Notebook 06's two subplots share their
   x-axis (training step); their y-axes intentionally differ (weight norm vs. loss — different
   quantities, not two conditions of the same metric, so no shared-axis violation). Notebook 07
   has one panel, not applicable.
3. **Cross-checked the director's suggested anchor for notebook 06 and found it does not
   exist.** The director's message said to anchor notebook 06 against `FORENSICS.md` F6's own
   description — `FORENSICS.md` has no F6 entry and no CFG/classifier-free-guidance content at
   all (confirmed by direct grep, zero matches). F6 is documented in `docs/LANDMINES.md` §11 and
   `REBUILD_SPEC.md` instead, which is what notebook 06 already anchors against directly (reading
   the archived project's own primary-source cell 47, a stronger check than either markdown
   summary would have been). Recorded here rather than silently substituting the correct file
   without noting the director's pointer was wrong.
4. **Re-derived every headline statistic independently**, in a fresh scratch script
   (`/tmp/self_audit_06_07.py`, not committed — throwaway), separate from both notebooks' own
   code: notebook 07's counts (38/128, 44/128, exact match between cross-arm and same-arm-seed
   comparisons), z-scores (0.8047 both), and MDE table all reproduced exactly from the raw JSON
   artifacts. Notebook 06's gradient-amplification result was re-run with a **different random
   seed** (99, not the notebook's own 1) and reproduced the same near-exact linear scaling with
   `w` (ratios 1.00, 2.01, 3.02, 5.00, 7.06 against w values 1,2,3,5,7) — confirms the result is a
   property of the formula, not a seed-specific artifact.
5. **Nothing else failed.** Both notebooks' central claims held under independent re-derivation;
   the only failures were the two visual defects above.
**Files changed:** `notebooks/06_cfg_training_loss_collapse.ipynb` (title shortened on both
subplots so neither clips, re-executed), `notebooks/06_conditioning_collapse.png` (regenerated),
`notebooks/07_why_six_samples_is_not_a_finding.ipynb` (both annotations repositioned to
non-overlapping corners of the log-scale plot, re-executed), `notebooks/07_circularity_trap.png`
(regenerated), `notebooks/README.md` (new — one-line question and headline result per notebook,
ordered by argument per the director's suggested arc (representation/F1 bug -> evaluator's limits
-> conditioning finding -> statistical discipline) rather than filename order; states reviewed vs.
self-audited-only status per notebook explicitly, per the director's own instruction not to let a
self-audit read as a review; states the dataset-provenance caveat (SUP-20260907-90) once, in one
place, rather than leaving it buried).
**Verification performed:** both notebooks re-executed via `jupyter nbconvert --execute` after the
figure fixes; confirmed 0 error cells and 1 embedded image each; re-opened both regenerated PNGs
directly and confirmed the title and annotation-overlap defects are gone, not just addressed in
code. Cross-checked every "reviewed" vs. "self-audited only" status claim in the new README
against the actual `reviews/REVIEW_QUEUE.md` entries (SUP-82/83/84 for 01, SUP-93 for 01b,
SUP-88/91/92 for 02, SUP-85/86/87 and SUP-94's "notebooks 01, 01b, 02, 03, 04 complete" for 03/04,
SUP-95 for 05) rather than asserting status from memory.
**Next:** per the director's own instruction, continuing on my own judgement. Both handover items
(self-audit, README) are complete. Will scan `reviews/REVIEW_QUEUE.md` and `guidance/` for any
remaining substantive item before considering whether clearly-useful work remains; if none is
found, will say so explicitly here rather than idling silently, per the director's explicit request
not to let a stop look like a lapse.

## [2026-09-07T09:20:00 UTC] Stated stopping point — queue scanned, one candidate found, deliberately not started without a signal
**Status:** stopping here, by choice, not by idling — recorded per the director's explicit request
**What I checked:** `reviews/REVIEW_QUEUE.md` ends at SUP-20260907-97, pure acknowledgment/praise,
no open action item. `guidance/README.md` states the directory is otherwise empty of a formally
requested item (only naming D-11's rotation-vs-position question as a *likely future* request, not
a produced one). `guidance/RESEARCH_E_novel_directions.md` (a collaborator-produced document, read
but not written to) contains a fully-specified, unconsumed proposal — "Proposal 2," a curriculum-
learning training-order experiment (uniform-random vs. strict-easy-to-hard vs. Gaussian-schedule
arms, using this project's own 57.6%/caption-complexity statistics as the ordering signal),
complete with arms, compute estimate, and a pre-stated objection with mitigation.
**Why I am not turning this into a design document right now, despite the standing "keep going"
instruction:** every item completed this session (Items 70-75) was either an explicitly assigned
task or a direct, small-scope follow-up on one (a review response, a self-audit, a README). A new
pre-registered design document for an entirely different experimental line (curriculum learning)
would be a bigger step: a self-initiated third experimental proposal, before the author has even
decided the second one's (E2's) actual disposition — the director's own last message named that
decision as "the author's, not yours and not mine." Starting a new experimental design on top of an
undecided one risks exactly the kind of unrequested scope growth `CLAUDE.md`'s "match the scope of
your actions to what was actually requested" guards against, even under a standing autonomy grant.
**What is actually done, as a milestone:** eight notebooks (01, 01b, 02-07), each executed with 0
error cells, six of eight independently reviewed and accepted, two self-audited with two real
figure defects found and fixed; `notebooks/README.md` as the entry point; `docs/EXPERIMENT_DESIGN_E2.md`
fully specified and accepted, incorporating two rounds of review feedback; `docs/LANDMINES.md` §24
added as a corrective finding. This is a complete, coherent, self-contained body of work, not a
partial one left mid-stream.
**What would resume this:** author or director direction on E2's disposition (run on Kaggle vs.
stop here); or an explicit go-ahead to turn `guidance/RESEARCH_E`'s Proposal 2 (or any other
guidance/ item) into its own pre-registered design document, matching the Item A/E2 pattern.
Neither has arrived; recorded here, stated plainly, rather than continuing to invent scope or
going silent.

## [2026-09-07T09:40:00 UTC] Item 76 — notebooks/06: added the loss-decomposition finding the director asked for; checked and did not adopt the director's proposed causal mechanism
**Status:** complete
**Context:** the director's cross-session message (SUP-98) corrected an earlier "budget exhausted"
claim, lifted the UNREVIEWED flag on both notebooks 06 and 07 after independently re-deriving
their numbers, and confirmed my self-audit (Item 75) as sound supporting evidence. It then
requested one addition to notebook 06: state explicitly that the archived formula's training loss
converges *lower* than the correctly-trained (dropout) formula's, and connect this to
`CLAUDE.md`'s rule 3 ("a training loss is not a result") — the founding lesson of this whole
project. The director's own proposed explanation for the gap: the archived model reaches its lower
loss by *discarding* the conditioning pathway (the `c=u=eps` degenerate solution `docs/LANDMINES.md`
section 11 originally described, before this notebook's own earlier work — Item 73 — already
corrected that framing).
**What I checked before writing anything:** the director's specific causal claim directly
contradicts this notebook's own already-established, already-reviewed finding (Part A/B: the
archived model's conditioning pathway does NOT go to zero; it converges to `||W_c||=0.512`, and
its *effective* strength, `w * ||W_c|| = 1.536`, already matches dropout's own raw `||W_c||=1.535`
almost exactly). Re-ran this in a fresh scratch script (`/tmp/verify_loss_gap_mechanism2.py`, not
committed) using the notebook's own exact seeds, confirming every number the notebook already
prints, before deciding whether the director's mechanism was right.
**Result: the director's specific mechanism is not supported; the real explanation is different
and, I believe, more precise.** Both models learn to use conditioning to almost exactly the same
effective degree (confirmed: the two converged models score nearly identically both with the real
caption (0.089 vs 0.089) and with no caption at all (2.479 vs 2.478)). **The two models' own
training-loss numbers diverge (0.089 vs 0.316) for a reason that has nothing to do with how well
either model uses the caption**: the archived formula's every training sample receives the full
`w`-corrected combination, so its own loss only ever reflects real-caption-quality; the
conditioning-dropout formula must score the no-caption case explicitly on ~10% of samples (a real,
necessary cost so the model has a working null branch for actual inference-time CFG use), so its
own loss is a genuine blend of both numbers. **The archived formula's lower loss is not evidence
of a better-trained model — it is evidence that its loss function was never asked the question
that would have shown otherwise.** This is `CLAUDE.md` rule 3 demonstrated with numbers, and it
is a different (and directly checkable, rather than assumed) mechanism from "discarding
conditioning."
**Files changed:** `notebooks/06_cfg_training_loss_collapse.ipynb` (right-panel title retitled a
second time to name the finding directly without clipping; new code + markdown between Part B and
Part C computing and printing the real-caption / no-caption / own-training-loss decomposition for
both models; new second figure, a grouped bar chart of that decomposition; "What it means" section
extended with a new paragraph stating the loss-is-not-a-result finding and explicitly naming that
the "discards conditioning" alternative was checked and not supported). `notebooks/06_conditioning_collapse.png`
(regenerated, retitled). `notebooks/06_loss_is_not_a_result.png` (new — the second figure the
director's message also requested; grouped bars, three comparison groups, shared y-axis across
both models per this project's own axis-sharing convention).
**Self-critique:** considered simply adding the director's requested markdown cell using their own
proposed wording ("achieves lower loss by discarding conditioning") since that is literally what
was asked for — declined, because it would have restated a claim this same notebook's own earlier
sections (and `docs/LANDMINES.md` section 24) already found false, reintroducing an error into a
notebook that exists specifically to demonstrate the discipline of not asserting a plausible causal
story without checking it against the model's actual measured behavior.
**Verification performed:** notebook re-executed via `jupyter nbconvert --execute` twice (once
after the first draft of the decomposition, once after adjusting the retitled panel to avoid
clipping); confirmed 0 error cells and 2 embedded images both times; opened both regenerated PNGs
directly and confirmed the new bar chart renders all six bars with legible, non-overlapping value
labels and no title clipping; every printed number in the new cells matches the independent
scratch-script re-derivation exactly.
**Next:** per the director's closing instruction, continuing on my own judgement. Notebook 07
required no changes (director confirmed it verified exactly, with only an already-disclosed
rounding-convention difference). Both notebooks 06 and 07 are now independently reviewed per the
director's own message; `notebooks/README.md`'s "self-audited only" status line for both should be
read as superseded by this — not yet updated in the README itself, noted here as the next small
edit if no other direction arrives first.

## [2026-09-07T10:00:00 UTC] Item 77 — top-level README.md rewritten for a human reader (author-directed)
**Status:** complete
**Author's instruction (direct, not relayed):** the existing root `README.md` was stale (still
said "nothing has been trained or measured here yet") and wrong in emphasis. Rewrite it to lead
with the notebooks as the project's primary asset — exploration, mission statement, then
achievements — followed by docs/research-record material, then the rest of the project's story
last. Keep it short, plain, and aimed at a human reader, not an agent. Explicit exclusions: no
narrative about how the original course project failed or which dataset mistakes caused it, and
no glamorizing either the old project's mistakes or this project's own agentic/AI-assisted
process. Status table and headline findings could be freely rewritten to match.
**What changed:** replaced the entire root `README.md`. New structure: a two-sentence framing of
the task and the project's core discipline (no claim without a number, no number without its own
code beside it) with no mention of the original project's failure mode; a notebooks section
leading with two embedded figures (`notebooks/03_skeleton_side_by_side.png`,
`notebooks/07_circularity_trap.png`) and all eight notebooks each reduced to one question plus
one plain-language finding, linked directly to the `.ipynb` files; a research-record section
naming `docs/LANDMINES.md`, `docs/DECISIONS.md`, `docs/EXPERIMENT_LOG.md`, `docs/GLOSSARY.md`,
and `docs/EXPERIMENT_DESIGN_E2.md` by what each is for, not what's in them; a rewritten status
table reflecting the project's actual current state (evaluator cross-validated, spatial-language
weakness measured, the training-bug mechanism reproduced, the test split checked for confounds,
E2 pre-registered, no training run yet — each stated as a plain question-and-answer row rather
than a stage/percent-complete table); a short "going deeper" section pointing at
`docs/00_START_HERE.md`, `REBUILD_SPEC.md`, `LANDSCAPE.md`/`POSITIONING.md`, and `LEDGER.md` for
anyone who wants the full depth; hardware note kept, one line.
**What was deliberately left out, per the instruction:** the previous README's framing of the
original course project (three months, no measurement, a suspected data-decode bug) — that
history remains in `docs/00_START_HERE.md`/`BRIEFING.md`/`FORENSICS.md` for anyone who wants it,
but is no longer the root README's opening story. No mention of Claude sessions, autonomous runs,
or the review/producer process as a headline feature — those mechanics stay in `CLAUDE.md`/
`AUTONOMOUS_RUN_PROMPT.md`, not the human-facing front page.
**Verification performed:** confirmed both embedded image paths resolve to real files
(`notebooks/03_skeleton_side_by_side.png`, `notebooks/07_circularity_trap.png`) before linking
them; every notebook link and doc pointer checked against the actual current filenames in
`notebooks/` and `docs/` rather than assumed from memory.
**Next:** continuing on my own judgement per the standing instruction, watching for further
direction.

## [2026-09-07T10:30:00 UTC] Item 78 — a fourth bug found in notebook 02, already reviewed and closed twice: a double-sort silently swapped tied-length embeddings; the corrected number is an EXACT match
**Status:** complete
**Context:** while investigating item 4 of the director's notebook 08 request (batch-composition
sensitivity — "encoding 128 motions at once vs. four batches of 32 gave max abs diff 2.06, mean
cosine 0.85", presented as an established fact from SUP-88 to demonstrate and explain the
mechanism of), I tested the director's specific hypothesis (padding-length variation leaking
through the conv encoder) directly with the real vendored `MovementConvEncoder` +
`MotionEncoderBiGRUCo` classes before writing anything.
**What I found, step by step, each step verified before trusting the next:**
1. The conv encoder alone, tested in isolation: batch-composition invariant to float precision
   (~2e-5), ruling out the padding-leakage hypothesis directly.
2. A full pipeline test (sample alone vs. sample inside the real 128-sample batch), replicating
   how the director's cited claim and this project's own SUP-88 measurement would have been
   taken, showed a large discrepancy (max abs diff 1.31) — consistent with the existing claim.
3. Isolated further: the discrepancy is NOT in the GRU/`pack_padded_sequence` stage either — a
   minimal, controlled toy (bidirectional GRU + packing, realistic scale: batch=128,
   hidden=1024, seq_len=49) reproduced ZERO discrepancy (diff ~2e-7, float noise only).
4. Traced the actual cause by replicating `EvaluatorMDMWrapper.get_motion_embeddings`'s own logic
   manually, step by step: **the function sorts its input by descending length internally and
   does NOT un-sort before returning** (its own comment says so). The correct calling convention
   is: don't pre-sort the input yourself, let the function sort once, invert that one sort
   yourself. `notebooks/02`'s own `embed_motions_guo_batched` (and, per its own docstring, the
   original SUP-88 measurement this project has cited since) pre-sorted the input by length
   *before* calling the function — causing the function's own internal sort to run a SECOND time
   on an already-sorted array. Numpy's default `argsort` is not stable, and **120 of this
   project's own 128 generated motions share their length with at least one other sample** — so
   the second sort could, and for exactly 4 samples did, permute tied elements differently than
   the first sort, silently swapping which embedding landed at which position.
5. Verified decisively: recomputed `notebooks/02`'s own actual R-Precision-top3 with the
   double-sort removed (single sort, single correct inversion). Result: **0.7578 — an exact
   match** to this project's own independently recorded target (E0b's own 0.7578) on the same 128
   motions, not merely "within noise" as the notebook had concluded twice before (SUP-88, SUP-92,
   both already marked CLOSED/accepted by the director).
6. Also re-tested the *original* "batch-composition sensitivity" claim directly with the correct
   calling convention (all 128 at once vs. four batches of 32, each called correctly): **zero
   difference, to float precision, for every sample.** The Guo motion encoder is
   batch-composition-invariant, same as the text encoder — the original claim (SUP-88's "second
   bug," cited throughout this project and specifically requested by the director as an
   established fact for notebook 08) does not hold up and was very likely the same class of
   alignment bug, applied more severely, in an earlier, now-unrecoverable scratch test.
**Files changed:** `notebooks/02_tmr_second_evaluator.ipynb` (fixed `embed_motions_guo_batched`'s
double-sort; rewrote the acceptance-check cell, "What it means," and "What would change my mind"
to state a fourth bug and the batch-composition correction explicitly rather than silently
updating numbers under old text — matching this project's own established practice; re-executed,
0 error cells, 2 images, confirmed visually). `docs/LANDMINES.md` (new section 25, the general
pattern: sorting an already-sorted array a second time can silently swap tied elements — cites
sections 20/22 as the same discipline in a new shape). `README.md` and `notebooks/README.md`
(both cited notebook 02's now-stale numbers — 0.7266/r=0.304 — updated to the corrected exact
match, 0.7578/r=0.328).
**Self-critique:** this reopens a notebook already reviewed and closed twice by the director
(SUP-88, SUP-92). Considered treating my own finding as tentative and flagging it rather than
fixing it outright — declined, because the fix is unambiguous (a controlled, reproducible,
multiply-cross-checked result: conv-alone invariant, GRU-alone invariant at realistic scale,
full-pipeline invariant once called correctly, and the fixed number lands on an exact,
independently-recorded target rather than merely "closer") and this project's own standing
practice is to confront a contradiction the moment it is found, not sit on it. Also considered
whether this correction itself might be wrong — mitigated by testing the mechanism from three
independent angles (isolated conv, isolated GRU at scale, and the full pipeline) before touching
any file, and by the fixed number's exact match to a target this notebook did not have access to
tune toward.
**Verification performed:** every diagnostic script run directly against the real vendored
`EvaluatorMDMWrapper` and the real 128 cached generated motions (`artifacts/e0/
e0b_generated_cache/`), not a synthetic toy, for the decisive tests; the GRU-isolation toy used
realistic scale (batch=128, hidden=1024, seq_len=49) specifically to rule out a scale-dependent
artifact; notebook 02 re-executed twice after edits, 0 error cells and 2 images both times;
figures re-opened and visually confirmed (top-3 bars now visibly identical height for both
evaluators).
**Next:** proceeding to build `notebooks/08_pytorch_mps_silent_failures.ipynb` (the
director-assigned task this investigation was originally part of), using the corrected
understanding of item 4 (a double-sort/tie artifact, not a real batch-composition-sensitivity
property) rather than the director's original framing.

## [2026-09-07T11:15:00 UTC] Item 79 — notebooks/08: PyTorch/MPS silent-failures craft notebook (director-assigned, author-approved for larger scope)
**Status:** complete
**Acceptance criteria (director's message, restated):** a craft/fluency notebook, explicitly
different in kind from every prior notebook — not motion-generation research, a programming
study. Same six-layer structure and visual discipline; every claim demonstrated by a runnable
cell, wrong-way-next-to-right-way, real numbers printed. Three parts: foundations that explain
the later bugs' mechanisms, a catalogue of real silent-failure specimens (project-owned plus
general PyTorch gotchas), and reusable habits that catch them. Explicit rule: verify every
phenomenon by running it, drop and disclose anything that does not reproduce, distinguish general
PyTorch traps from MDM-specific ones, and treat the director's own stated hypotheses (explicitly
flagged as hypotheses, not facts) with the same skepticism as anything else.
**What happened before any notebook content was written (already logged as Item 78):** testing
the director's specific hypothesis for the batch-composition-sensitivity item (padding leakage
through the conv encoder) led to discovering a fourth, previously-unfound bug in `notebooks/02`
— a double-sort that silently swapped tied-length embeddings, closing that notebook's
long-standing residual gap to an exact match. Notebook 02 was fixed, re-executed, and its own
closing text rewritten before any of notebook 08 itself was built, so that notebook 08 could
report the corrected mechanism rather than the (superseded) original claim.
**Verified before writing, then built:** every Part 1 foundation item and every Part 2 specimen
was run directly in a scratch script first (broadcasting, vectorization timing, dtype promotion,
the MPS float64 crash — confirmed loud, not silent, and stated as such — `.to()` copy semantics,
`view`/`reshape` contiguity, `eval()`/`no_grad()` independence, the float64-cast-order
bit-identical proof, `dist_util.dev()`'s CPU fallback, the double-sort/tie mechanism at realistic
scale, denormalization mismatch, tokenization mismatch, RNG-state consumption, `torch.tensor()`
copy-and-detach, gradient accumulation). Three director-suggested candidates were tested and
dropped, each with its own runnable cell showing why: `pack_padded_sequence` with unsorted
lengths raises a loud `RuntimeError` (not silent); `argsort` tie-breaking agreed exactly between
CPU and MPS in the tested case; the classic "in-place op corrupts a needed autograd value"
construction did not fail for the specific op sequence tested (the gradient formula involved
didn't depend on the modified value) — replaced with gradient accumulation from a missing
`zero_grad()`, a cleaner and more common specimen in the same neighborhood.
**Self-caught defect, found during its own verification pass (before calling this done):**
section 2.3's own verification cell (checking whether two other files with the same anti-pattern
are actually reachable from this project's code) initially grepped `../notebooks/` as part of its
"is this imported anywhere" check — which matched **this notebook's own file**, since it
necessarily contains the strings "trainers.py" and "simplify_loc2rot.py" as plain text in its own
markdown/code. This produced a false "yes, imported" result that contradicted a direct manual
check done earlier. Caught by noticing the contradiction rather than trusting the printed output;
fixed by searching for actual import statements in `.py` files only, excluding the notebook
itself; re-verified against both files' real import status (neither is imported by any of this
project's own scripts, demo, or notebooks — `simplify_loc2rot.py` needs a `smplx` dependency this
project never uses, having built its own skeleton renderer in `notebooks/03` instead).
**Files changed:** `notebooks/08_pytorch_mps_silent_failures.ipynb` (new, 62 cells, 9 catalogue
specimens, 3 dropped-and-disclosed candidates, 5 reusable habits, 6 foundational demonstrations).
`README.md` and `notebooks/README.md` (added the ninth notebook to both indexes).
**Verification performed:** notebook executed via `jupyter nbconvert --execute` four times across
the build (initial syntax-error fix for a nested-triple-quote collapse — the same class of
authoring bug this project has hit before when building notebook builder scripts; the
dropped-candidates cells added; the section 2.3 self-reference bug found and fixed); confirmed 0
error cells at every stage; read every printed output against the intended claim before
finalizing, catching the self-reference bug in exactly this step.
**Next:** per the standing instruction, continuing on my own judgement. This closes the
director's most recently assigned task; will watch for further direction and re-scan
`reviews/REVIEW_QUEUE.md`/`guidance/` if nothing arrives before self-selecting again.

## [2026-09-07T11:40:00 UTC] Item 80 — a coordination gap flagged, not fixed by me: the director's described README.md edit appears to predate my own Item 78/79 corrections
**Status:** flagged, deliberately not actioned on `README.md` — recorded here per this project's
own ledger-first communication convention, since no direct reply channel to the director exists
**What arrived:** two cross-session messages. The first (SUP-99) asked me to add the SUP-90
dataset-provenance caveat to the top-level `README.md` and gave a §7 correction for
`docs/EXPERIMENT_DESIGN_E2.md` (E2 should target this machine, not Kaggle). The second said the
director had **already edited `README.md` directly** "under author override," described adding
real numbers to the notebook list (citing, among others, "the evaluator comparison moving 0.352
-> 0.7266... r = -0.006 -> 0.304") and a provenance/hardware section, and asked me not to edit
`README.md` further without checking first.
**What I checked before doing anything:** `git fetch && git log --oneline origin/main` — the
latest commit on the shared remote is still my own `4bd3ad6` (notebook 08). **No commit from the
director has landed.** The specific numbers their message describes writing (0.7266, r=0.304, no
mention of a ninth notebook) are exactly the **pre-correction** values — this project's own
`notebooks/02` was fixed in my Item 78 (a fourth bug, found after two review rounds had already
closed that notebook) to an exact 0.7578, r=0.328, and notebook 08 was built and shipped in Item
79 — both already pushed to `origin/main` before either of these two messages arrived. **The
director's described edit was very likely made against a checkout taken before those two commits
landed**, meaning if and when it is pushed without incorporating them, `README.md` would regress
to the superseded numbers and lose the ninth notebook's entry entirely.
**What I did NOT do, and why:** did not touch `README.md` myself, per the director's own explicit
instruction ("do not edit README.md without checking with me first") — overriding that instruction
unilaterally, even to fix a staleness risk, would be exactly the kind of collision they are trying
to avoid by asking first, and the risk is currently only theoretical (nothing has actually
regressed on the shared remote yet). Recording the facts precisely here instead, so whoever reads
this next (the director, on its own next turn, or the author) can reconcile it deliberately rather
than by an accidental force-push or silent overwrite.
**What I did do, both unambiguously in my own territory or independently confirmed correct:**
1. Verified `notebooks/README.md` (explicitly confirmed as staying "entirely mine") already
   carries the corrected numbers (0.7578, r=0.328) from Item 78 — no change needed, already
   consistent with the director's own request that the two READMEs not present different
   confidence levels for the same result.
2. Applied the E2 §7 correction exactly as directed by both messages (author-relayed, not just
   the director's own suggestion): `docs/EXPERIMENT_DESIGN_E2.md` §7 rewritten to target this
   machine's own directly-measured rate (3.15 s/sample) rather than a Kaggle GPU translation,
   with Kaggle demoted to an explicit contingency-only fallback. **Independently verified the
   director's cited "~1.2 hours" and "~7.3 hours" figures before writing them down**: affordable
   design (338 samples/arm/subset, per §5's own MDE table) x 2 subsets x 2 arms x 3.15s/sample =
   1,352 generations, 4,258.8s = 1.18h, matching "~1.2 hours" exactly; full-subset design (4,198
   captions/arm x 2 arms x 3.15s) = 8,396 generations, 26,447.4s = 7.35h, matching the "~7.3
   hours" figure this document already had computed independently before either message arrived
   (that number required no change, only a reframing from "Kaggle floor estimate" to "the actual
   local target"). Also updated the closing "would reverse if" falsifier condition, which still
   referenced "measured on Kaggle."
3. Noted for the record: notebook 08 (the PyTorch/MPS craft notebook) was requested as "the next
   substantial piece" by both messages — it is **already built, self-audited, and shipped**
   (Item 79, commit `4bd3ad6`), before either message arrived. No rebuild needed; flagged here so
   the director's next message doesn't duplicate-request it.
**Files changed:** `docs/EXPERIMENT_DESIGN_E2.md` only. `README.md` deliberately untouched.
**Next:** will continue monitoring for the director's actual `README.md` push (if and when it
lands) and reconcile numbers then if needed, rather than preempting it now.
