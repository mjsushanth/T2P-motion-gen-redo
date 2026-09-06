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

