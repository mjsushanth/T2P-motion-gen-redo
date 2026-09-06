# LEDGER — T2P Reboot Forensics (append-only, timestamped)

## 2026-09-05 22:10 — Session start
- Read BRIEFING.md in full. Source project confirmed read-only at
  `/Users/joel/MJS_ROOT/MJS_STUDY/<ARCHIVE>/`.
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
**Next:** Sent a status message to the peer review session (`local_b04c6f9d-60e0-4bc7-a074-92669bab6d19`,
"T2P diffusion Opus Review") summarizing corrections applied, housekeeping done, and the repo
naming decision, and stating I'll flag disagreements rather than execute Stage 2 verbatim.
Starting Stage 2 Part A (landscape survey) now via parallel research passes.

## [2026-09-06T07:05:00] Item 3 — Read project scaffold; corrected two stale LANDMINES.md entries
**Status:** complete
**Acceptance criteria:** read `docs/00_START_HERE.md`, `docs/LANDMINES.md`, `docs/DECISIONS.md`,
`docs/CODE_MAP.md`, `docs/EXPERIMENT_LOG.md` per the peer review's R-2 before starting Stage 2
work; note `docs/DECISIONS.md` D-17 already reflects Joel's "full redo, not eval-first" steer and
promotes positioning to a first-class `POSITIONING.md` deliverable (not a REBUILD_SPEC.md
appendix as Stage 2's original instructions said) — planning around the updated D-17, not the
original instruction text.
**Files changed:** `docs/LANDMINES.md` §5 (cluster mismatch) and §4 (loss-reduction claim), both
corrected to match the peer review's R-1 findings, which had been folded into `FORENSICS.md`'s
OPEN_QUESTIONS but not yet propagated to `LANDMINES.md`, which still stated the old (wrong)
attributions.
**Environment changes:** none.
**Self-critique defects found:** `LANDMINES.md` §5 said "the paper and the sampling config say
8" — wrong; both are 10, and "8" is docs-only (README.md, Obsidian notes). §4 attributed the
"99.995% loss reduction" phrasing to the PDF report's Table 1 — wrong; the PDF has only the raw
loss values, the phrase itself is in the Obsidian interview-prep notes. Both were stale because
they were written before the peer review's R-1 correction landed in `FORENSICS.md`, and nobody
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

## [2026-09-06T07:15:00] Item 4 — Licence check on vendored primary_source/ files (urgent, per peer review)
**Status:** complete
**Acceptance criteria:** the peer review session flagged that `primary_source/` (files from
`github.com/EricGuo5513/HumanML3D`, vendored for F1 verification) is now in a **public** repo,
which makes its licence status load-bearing rather than a footnote. Confirm the upstream licence
permits redistribution-with-modification, and add whatever attribution/LICENSE file it requires,
before doing anything else in Stage 2.
**Files changed:** `primary_source/LICENSE` (added — verbatim upstream MIT license text, fetched
directly, not retyped from memory). `primary_source/PATCHES.md` (added a Licence section).
**Environment changes:** none.
**Self-critique defects found:** the vendored files had been sitting in a now-public repo without
their required copyright/permission notice since the push in Item 2 — a real (if low-severity,
MIT is a permissive licence) compliance gap that I did not catch myself; the peer review caught
it.
**Verification performed:** `GET https://api.github.com/repos/EricGuo5513/HumanML3D/license`
returns `license.key: "mit"`; fetched the actual `LICENSE` file content (not just trusting the
GitHub-detected key) via `raw.githubusercontent.com/EricGuo5513/HumanML3D/main/LICENSE` — MIT
License, Copyright (c) 2022 Chuan Guo, standard text, saved verbatim to `primary_source/LICENSE`.
MIT requires the copyright+permission notice be included in copies/substantial portions; adding
the LICENSE file alongside the vendored files satisfies that.
**Next:** apply the same discipline to anything Stage 2 proposes vendoring later (PoseScript,
SMPL/SMPL-X, evaluator code) — SMPL-X specifically is registration-gated per the peer review;
this is now item 1 of the Part A licence survey, not a footnote, per their message.

**Note, no action taken:** the peer review session separately flagged to Joel directly (not to
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

## OPEN_QUESTIONS

- **D-11/12/13 status.** Argued in full in `REBUILD_SPEC.md` (dataset: switch to full-sequence
  HumanML3D, not static pose; representation: redundant vector baseline + rotation/FK ablation;
  text encoder: CLIP-token-level baseline + DistilBERT ablation). Left PENDING in
  `docs/DECISIONS.md` pending the director session's review, per our agreed process — this is
  the one deliberate incompleteness in an otherwise "complete" Stage 2.
- **Task-reframing proposal** (`REBUILD_SPEC.md` §0): switching the primary research task from
  static single-pose to full text-to-motion sequences is a real disagreement-in-waiting with the
  original Stage 2 brief's framing (which asked to resolve "PoseScript vs corrected HumanML3D,"
  both static-pose). Flagging prominently rather than quietly substituting one task for another.

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

