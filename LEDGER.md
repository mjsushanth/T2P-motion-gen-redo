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

## OPEN_QUESTIONS

- **Should this session proceed to Stage 2 now?** A peer review session has reviewed and accepted
  this session's Stage 1 forensics output and asked it to begin Stage 2 (web-research landscape
  survey + `REBUILD_SPEC.md`, resolving `docs/DECISIONS.md` D-11/D-12/D-13) immediately. This
  session's own originating instructions explicitly said "Do not proceed past forensics. Stop and
  report," and several other peer sessions are currently active on this same project directory —
  it is not established whether one of those is already tasked with Stage 2, or whether Joel wants
  this session specifically to continue. Holding here pending Joel's direction rather than
  guessing.

