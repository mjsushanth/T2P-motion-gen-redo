# T2P Reboot — Forensic Briefing (Opus 5, 2026-09-05)

Source project (READ-ONLY, do not modify):
`/Users/joel/MJS_ROOT/MJS_STUDY/<ARCHIVE>/`
- `DL_T2P_IMPL.ipynb` (87 cells; implementation lives in 4 monolithic cells: #13, #31, #41, #47 — 44k/46k/52k/101k chars)
- `DL_T2P_V2 (EDA).ipynb` (48 cells)
- `the original project report (PDF)` — NeurIPS-format report, 9 pages
- `README.md`, `RESEARCH_README.md`, `DESIGN_README.md`, `dl_t2p_proj_0321.yml` (win-64 + CUDA 11.8 conda env)

Obsidian notes (READ-ONLY):
`/Users/joel/Library/Mobile Documents/iCloud~md~obsidian/Documents/Study_Notes_Obsd/091 AI Coursework - Project Deepdives/`
- `DL - T2P Deep Dive.md` (86 KB, 13 sections)
- `DL - T2P Experience MM.md` (TOC only)

Host: macOS / Apple Silicon. Original env is Windows+CUDA — **not reproducible here**.

---

## What the project did

Text -> single static 3D human pose. HumanML3D (HF: `TeoGchx/HumanML3D`), 23,384 train samples.
Take the **first frame** of each motion sequence as the "pose"; take the **first action** of the
caption (truncate at first CCONJ/SCONJ) as the text. Encode text with **frozen CLIP ViT-B/32
global 512-d embedding** -> learned 512->256 projection. Denoise a 66-d vector with a
**1-D UNet (96/192/384) + multi-head cross-attention + conditioned residual blocks**, DDPM
(1000 steps, squaredcos), classifier-free guidance ramped 2.0 -> 7.0, plus a bone-length
"anatomy loss" and post-hoc forward-kinematics bone-length clamping. ~2.1M trainable params.

Reported outcome (Table 1 of the paper): Phase 1 final loss 1.52e15 -> Phase 2 1.17 -> Phase 3 0.69.

---

## Findings (confidence labelled; VERIFY BEFORE BUILDING ON THESE)

### F1 — The 66-dim slice is almost certainly the wrong 66 dims. [HIGH CONFIDENCE / EMPIRICALLY UNVERIFIED]
`HumanML3DProcessor.extract_pose` does `motion_sequence[frame_idx][:66]` on a `[T, 263]` array
and then reshapes to `(22, 3)`, treating index 0 as the pelvis.

The canonical HumanML3D 263-d layout (Guo et al. 2022, `motion_representation.py`) is:
```
[0]       root angular velocity          (1)
[1:3]     root linear velocity x,z       (2)
[3]       root height y                  (1)
[4:67]    ric_data — local positions of the 21 NON-root joints   (63)
[67:193]  rot_data — 6D rotations, 21 joints                     (126)
[193:259] local velocities, 22 joints                            (66)
[259:263] foot-contact flags                                     (4)
                                                        total = 263
```
So `[:66]` = 4 root scalars + the first 62 of 63 ric values. Reshaped to `(22,3)`, "joint 0"
is actually `[root_rot_vel, root_vel_x, root_vel_z]` — a velocity triple, not a position —
and **every subsequent joint is shifted by 4 scalars**, mixing X/Y/Z across joint boundaries.

The EDA notebook independently *hypothesised* a layout of `66 pos + 66 vel + 126 rot + 5 = 263`
from correlation heatmaps. That also sums to 263, which is why the error was never caught.

If F1 holds, it is the single root cause behind the whole documented struggle:
- the "extreme left hip offsets" and "spine joints near the floor" anomalies
- the "coordinate system mystery" (bodies face +Y, left on +X) — a rationalisation of a
  misalignment artifact, not a real canonicalisation
- unstable bone lengths, hence the need for a reference-bone-length DB + FK clamping
- Z-axis degeneracy, left/right confusion, "anatomically plausible but semantically random"

Also note: even the correct `ric_data` slice is root-relative *local* data. The intended decode
is `recover_from_ric()` from the HumanML3D repo, which integrates root rotation/translation.

**How to verify (do this first, ~30 min):** load 200 samples, slice `[:66]`, reshape `(22,3)`,
compute the 21 bone lengths using the documented skeleton and report mean/std across samples.
Then do the same with `recover_from_ric(motion, 22)`. Correct decode => bone lengths near-constant
per bone across samples and frames (std/mean < ~5%). Misaligned decode => high variance. Report
both tables side by side. If bone lengths are near-constant under `[:66]`, F1 is REFUTED — say so
loudly and stop.

### F2 — Cluster-aware sampling was fit on different data than it sampled. [HIGH CONFIDENCE]
EDA `extract_static_poses` clusters poses taken from a **random frame** per sequence, on the raw
un-normalised `[:66]` slice (which under F1 includes root height and root velocity, so K-means is
partly clustering *how fast the root is moving*). Training then uses the **first frame** of the same
sequences with those cluster labels. The labels do not describe the vectors being trained on.
**CORRECTED 2026-09-05 (Opus 5, after C1's Stage 1):** the briefing originally said "EDA
`main()` uses `n_clusters=10`; the paper and sampling config use 8." That framing was wrong.
Both notebooks use **10** consistently, as C1 verified. The "8" comes from the *documentation*
— `README.md` ("8-cluster balanced sampling strategy") and the Obsidian deep dive ("8 pose
clusters (K-means)") — never from the code. The PDF states no cluster count at all. So the
discrepancy is real but it is **docs-vs-code**, not EDA-vs-IMPL. Resolves C1's OPEN_QUESTION 1.

Two further docs-vs-code contradictions found while resolving this, both VERIFIED:
- The PDF (§1.1.2) says "K-means clustering was then performed **on the reduced embeddings**."
  The code fits KMeans on the **raw 66-d** vectors (`cluster_poses(poses)` where `poses` is the
  un-reduced slice). The PCA/t-SNE outputs were used for plotting only.
- `README.md` claims the sampling "avoided 49.6% cluster dominance." C1 measured cluster 0 at
  **38.77%** directly from `clusters.npy`. The 49.6% figure is not reproducible from the saved
  artifacts.

### F3 — The task framing caps achievable performance regardless of architecture. [HIGH CONFIDENCE]
Pairing "the first action of the caption" with "frame 0 of the sequence" is a near-systematic
mismatch. Frame 0 of "a person walks forward" is a neutral standing pose; frame 0 of "a person
does a cartwheel" is also a neutral standing pose. A large fraction of training pairs therefore
teach (action text -> rest pose). The model cannot learn a mapping the data does not contain.
This is the deeper conceptual problem and is independent of F1.

### F4 — There are no results. [VERIFIED by reading]
No validation loop, no test-split evaluation, no seeding (`torch.manual_seed` absent), no
experiment tracking (no wandb/mlflow/tensorboard). The only numbers reported anywhere are
**training losses**. Table 1 compares "Final Loss" across three phases whose loss functions are
different objects (Phase 1's 1.5e15 is a bug artifact, not a measurement), so the
"99.995% loss reduction" framing is not a valid comparison.

**CORRECTED 2026-09-05 (Opus 5):** the briefing originally called that the *paper's* headline.
It is not. The string "99.995" appears **zero times** in `the original project report (PDF)`
(VERIFIED by `pdftotext` + grep). The PDF reports only the raw Table 1 values. The 99.995%
framing appears five times in the **Obsidian deep dive** — including in the scripted interview
answer at line 3139. That makes it a *study-notes* claim rather than a published one, which
matters: the invalid comparison is the one Joel has been rehearsing for interviews. Resolves
C1's OPEN_QUESTION 4 for this specific number.

None of the field-standard HumanML3D metrics were computed: R-Precision (top-1/2/3), FID,
MM-Dist, Diversity, MultiModality — nor any pose metric (MPJPE / PA-MPJPE) or bone-length error.
**Consequence: the project's results were never actually subpar. They were never measured.**

### F5 — Engineering state. [VERIFIED]
Four notebook god-cells hold the entire system; three near-duplicate re-implementations of the
same classes coexist in one file (`ResidualBlock`, `UNetModel`, `DiffusionTrainer` defined 3x).
No package, no config objects, no tests, no CLI, no seeds, hardcoded Windows paths
(`r"Iter7_18300p_full_210425\dataset\..."`). Env yml is 12 KB of win-64 build-pinned strings.

---

## Guardrails for the worker

- The source project directory is **read-only**. Never edit, move, or delete anything under
  `<ARCHIVE>/`. Copy out what you need.
- **No `conda create`, `pip install`, `uv pip install`, or `brew install` without asking Joel first.**
  Write the environment file, print the exact command, and stop for approval.
- No git commits. No downloads of large datasets without asking (state the size first).
- Absolute imports only. No `sys.path` hacking, no `parent.parent` chains.
- No emojis in code. 4-space indent. Type hints on function signatures.
- Label every claim VERIFIED or UNVERIFIED. Never invent a metric, a URL, a paper result, or an
  API signature. "Unknown" beats a plausible guess.
- Append every finding to `T2P-Reboot/LEDGER.md` as you go (append-only, timestamped).
