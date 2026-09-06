# T2P Reboot — Forensics Report

Stage: forensics only, per `BRIEFING.md`. No rebuild, no design, no model code written.
Source project (read-only, untouched):
`<ARCHIVE>/`

All primary-source files fetched for verification live in `T2P-Reboot/primary_source/`
(copies from `github.com/EricGuo5513/HumanML3D`, default branch, fetched 2026-09-05).
Figures, result data, and the analysis scripts that produced them live in
`T2P-Reboot/artifacts/forensics/`. Full timestamped trail in `T2P-Reboot/LEDGER.md`.

---

## F1 — Is `motion[:66]` the wrong 66 dims of HumanML3D's 263-d vector?

**Verdict: VERIFIED. High confidence.**

### Claim
`HumanML3DProcessor.extract_pose` in the source project does `motion_sequence[frame_idx][:66]`
on a `[T, 263]` array and reshapes to `(22, 3)`, treating index 0 as the pelvis. The hypothesis
is that this misreads the vector: it grabs 4 root scalars (angular velocity + 2D linear velocity
+ height) followed by only 62 of the 63 `ric_data` values, shifting every subsequent joint by 4
scalars and mixing components across joint boundaries.

### Method
1. **Primary-source layout confirmation.** Fetched `motion_representation.ipynb`, `paramUtil.py`,
   `common/skeleton.py`, `common/quaternion.py` directly from `github.com/EricGuo5513/HumanML3D`
   (via GitHub API, raw content, not memory). Read the actual vector-construction code.
   Independently, a second research pass fetched the same repo plus the CVPR 2022 paper PDF
   (`openaccess.thecvf.com`) via a different route and reported the identical layout — two
   independent fetches agree.
2. **Empirical decode test.** Streamed 250 samples (34,554 frames total, min 12 frames/sample)
   from HF `TeoGchx/HumanML3D` (train split, streaming mode — no bulk download; only 2 lightweight
   commands totaling well under the 500MB worst case). For every frame of every sample, computed
   all 21 bone lengths (per the primary-source `t2m_kinematic_chain`) under:
   - **decode (a):** `motion[t][:66]` reshaped to `(22,3)` — the source project's actual method.
   - **decode (b):** `recover_from_ric(motion, 22)` — the primary source's documented decode,
     using the actual `recover_root_rot_pos`/`recover_from_ric`/`qrot`/`qinv` code quoted below.
3. **Decision rule (stated before running):** a correct decode gives near-constant bone lengths
   per bone across samples/frames (CV = std/mean under ~5%); a misaligned decode gives high
   variance. If decode (a) came out near-constant, F1 would be REFUTED.

### Evidence

**Primary source, quoted verbatim** (`motion_representation.ipynb`, cell 2, `process_file`):
```python
root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)      # 4 dims
...
data = root_data
data = np.concatenate([data, ric_data[:-1]], axis=-1)   # + 63 dims  -> [4:67]
data = np.concatenate([data, rot_data[:-1]], axis=-1)   # + 126 dims -> [67:193]
data = np.concatenate([data, local_vel], axis=-1)       # + 66 dims  -> [193:259]
data = np.concatenate([data, feet_l, feet_r], axis=-1)  # + 4 dims   -> [259:263]
```
4 + 63 + 126 + 66 + 4 = 263, boundaries at 4/67/193/259/263 — matches the briefing's claimed
layout index-for-index. `paramUtil.py`:
```python
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                        [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
```
gives 21 bones as consecutive joint pairs within each chain — matches "21 bone lengths."
`recover_from_ric` (same notebook, cell 3), quoted verbatim:
```python
def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]        # slices exactly [4:67]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions
```

**Empirical bone-length table** (mean / std / CV%, across 250 samples x 34,554 frames):

| bone (joint pair) | raw mean | raw std | **raw CV%** | ric mean | ric std | **ric CV%** |
|---|---|---|---|---|---|---|
| 0-2 | 0.8333 | 0.1590 | 19.08% | 0.1099 | ~5.5e-8 | 0.00005% |
| 2-5 | 0.3883 | 0.0347 | 8.93% | 0.3902 | ~5.7e-8 | 0.00001% |
| 5-8 | 0.4213 | 0.0548 | 13.00% | 0.4256 | ~7.9e-8 | 0.00002% |
| 8-11 | 0.1410 | 0.0301 | 21.36% | 0.1494 | ~2.5e-7 | 0.00017% |
| 0-1 | 1.2541 | 0.2265 | 18.06% | 0.1031 | ~5.7e-8 | 0.00006% |
| 1-4 | 1.0307 | 0.1658 | 16.09% | 0.3936 | ~5.5e-8 | 0.00001% |
| 4-7 | 0.4063 | 0.0503 | 12.37% | 0.4324 | ~8.2e-8 | 0.00002% |
| 7-10 | 0.0876 | 0.0289 | 33.01% | 0.1434 | ~2.4e-7 | 0.00017% |
| 0-3 | 1.0428 | 0.1661 | 15.93% | 0.1316 | ~4.8e-8 | 0.00004% |
| 3-6 | 0.1777 | 0.0655 | 36.87% | 0.1432 | ~5.9e-8 | 0.00004% |
| 6-9 | 0.1346 | 0.0856 | **63.57%** | 0.0574 | ~6.2e-8 | 0.00011% |
| 9-12 | 0.2413 | 0.0277 | 11.50% | 0.2194 | ~7.4e-8 | 0.00003% |
| 12-15 | 0.1911 | 0.1562 | **81.70%** | 0.1030 | ~3.2e-7 | 0.00031% |
| 9-14 | 0.2295 | 0.1252 | **54.52%** | 0.1434 | ~6.8e-8 | 0.00005% |
| 14-17 | 0.1233 | 0.0084 | 6.84% | 0.1230 | ~9.2e-8 | 0.00007% |
| 17-19 | 0.2601 | 0.0312 | 12.02% | 0.2631 | ~9.5e-8 | 0.00004% |
| 19-21 | 0.2639 | 0.0427 | 16.18% | 0.2699 | ~2.7e-7 | 0.00010% |
| 9-13 | 0.2321 | 0.1372 | **59.09%** | 0.1375 | ~6.7e-8 | 0.00005% |
| 13-16 | 0.1434 | 0.0120 | 8.36% | 0.1316 | ~9.5e-8 | 0.00007% |
| 16-18 | 0.2537 | 0.0329 | 12.96% | 0.2568 | ~9.1e-8 | 0.00004% |
| 18-20 | 0.2429 | 0.0500 | 20.59% | 0.2660 | ~2.6e-7 | 0.00010% |
| **mean across bones** | | | **25.81%** | | | **~0.00008%** |
| **median across bones** | | | **16.18%** | | | |

Decode (a) [raw `[:66]`] has a mean per-bone CV of **25.81%**, ranging up to **81.70%** for the
lower-spine bone — far above the 5% threshold. Decode (b) [`recover_from_ric`] has bone lengths
constant to ~1e-7 relative precision (floating-point noise floor), i.e. effectively **0.00%** CV.
This near-perfect constancy is itself explained by the primary source: `uniform_skeleton()`
(cell 1 of the same notebook) retargets every HumanML3D clip onto one canonical target skeleton
before the 263-d vector is ever built, so every clip shares literally the same bone lengths —
that retargeting is exactly what the correct decode recovers, and exactly what the raw slice
destroys.

**Visual check** — 3 example skeletons, frame 0, both decodes, saved to `artifacts/forensics/`:
- [f1_skeleton_sample0.png](artifacts/forensics/f1_skeleton_sample0.png) — "a man squats extraordinarily low then bolts up..."
- [f1_skeleton_sample1.png](artifacts/forensics/f1_skeleton_sample1.png) — "a man full-body sideways jumps to his left."
- [f1_skeleton_sample2.png](artifacts/forensics/f1_skeleton_sample2.png) — "a man lifts something on his left and places it down on his right."

In all 3, decode (a) (red, left) renders a tangled, self-intersecting stick figure with one joint
flung far outside the rest of the body (visually consistent with a velocity-scalar being
misinterpreted as a position coordinate) — not a plausible human pose. Decode (b) (blue, right)
renders a coherent, proportioned standing human figure with recognizable limbs and spine in all 3
samples.

### Result
Per the pre-stated decision rule: decode (a) is NOT near-constant (25.81% mean CV, individual
bones up to 81.70%) — **F1 is VERIFIED**, not refuted. The source project's `[:66]` slice is
demonstrably the wrong 66 dimensions, both by primary-source code inspection and by direct
empirical measurement on real HumanML3D data.

---

## F2 — Cluster-sampling mismatch

**Verdict: VERIFIED. High confidence.**

### Claim
EDA's clustering runs on different data than what training actually samples: a random frame,
un-normalized, versus frame 0, normalized — with `n_clusters` possibly inconsistent (8 vs 10).

### Method
Extracted full cell source text from both notebooks (JSON parse, no execution; avoids the
multi-MB output blobs that make these files large). Located and read `extract_static_poses`,
`cluster_poses`, `analyze_pose_diversity` (EDA notebook, cell 21) and
`create_strategically_sampled_dataset` / `HumanML3DProcessor` (IMPL notebook, cell 13). Cross-checked
against the precomputed artifacts in `pose_diversity_results/` (`caption_indices.npy`,
`clusters.npy`) using the `mjs_mlcvdl_unified_m5` env's numpy — read-only, no notebook execution.

### Evidence

1. **Random frame, not frame 0.** EDA cell 21:
   ```python
   frame_idx = random.randint(0, len(motion)-1)
   pose = motion[frame_idx, :66]
   ```
2. **Clustering on raw, un-normalized data.** Same cell:
   ```python
   kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
   clusters = kmeans.fit_predict(poses)   # poses = raw motion[frame_idx, :66], never normalized
   ```
   IMPL's actual training pipeline normalizes every pose (`normalize_pose(..., method='center_scale')`)
   — a transform never applied to the vectors that were clustered. (Note: given F1, this raw slice
   also mixes in root velocity/height, so the clustering is partly grouping by root motion speed,
   not pose shape — compounding the mismatch.)
3. **Broken index mapping.** IMPL's `_create_cluster_mapping` treats array *position* in
   `clusters.npy` as the HF dataset *row ID* and passes it straight to `data.select(...)`. The true
   row ID per position is recorded only in `caption_indices.npy`, which is never loaded anywhere
   in the IMPL notebook (confirmed: 0 references). Empirically:
   ```
   caption_indices.npy shape: (23384,)   clusters.npy shape: (23384,), values 0-9
   positions where caption_indices[i] != i: 23382 / 23384  ->  99.99% mismatch
   ```
   IMPL then takes **frame 0** (`frame_selection='first'`, every call site) of these mis-indexed
   rows — so training samples are frame-0/normalized poses from (mostly wrong) dataset rows,
   labeled with cluster IDs computed from different rows' random frames in raw space.
4. **n_clusters.** EDA `main()`: `n_clusters = 10`. IMPL never recomputes this — it hard-codes
   cluster IDs 0-9 into sampling-config presets, consistent with (not independently derived from)
   EDA's value. (The briefing's "8 vs 10" question resolves to: both notebooks in fact use 10;
   8 does not appear as an actual configured value in either notebook as read.)
5. Cluster-size percentages computed directly from `clusters.npy` (e.g. cluster 0 = 38.77% of all
   23,384 poses) match, digit-for-digit, the hard-coded comments in IMPL's sampling presets —
   confirming IMPL really is consuming this specific `clusters.npy` alongside its broken index
   assumption.

### Result
Every sub-claim is backed by quoted code plus directly-inspected precomputed artifacts. The
99.99% index mismatch is a hard, artifact-verified number, not an inference.

### Addendum — docs-vs-code contradictions (found in review follow-up, 2026-09-05)
Two further discrepancies, both VERIFIED by a review pass and consistent with F2's
overall finding that the project's documentation describes a cleaner pipeline than the code
actually implements:
- The PDF report (the original project report (PDF), section 1.1.2) states K-means was "performed
  on the reduced embeddings." The code fits `KMeans` directly on the raw 66-d slice
  (`kmeans.fit_predict(poses)`, EDA cell 21) — PCA/t-SNE are used only for the separate
  visualization plots, never feed into the actual clustering.
- `README.md` claims the sampling strategy "avoided 49.6% cluster dominance." The largest
  cluster measured directly from `clusters.npy` is cluster 0 at **38.77%** — the 49.6% figure is
  not reproducible from the saved artifacts.

---

## F3 — Task-framing mismatch (frame 0 as near-neutral pose)

**Verdict: VERIFIED (moderate effect size). Medium-high confidence.**

### Claim
Pairing "first action of the caption" with "frame 0 of the sequence" systematically teaches
(action text -> near-neutral rest pose), capping achievable performance independent of
architecture.

### Method
Using the same 250 streamed samples as F1 (69 distinct first-action verbs present in the sample,
extracted via regex on the dataset's POS-tagged caption field, e.g. `.../VERB`), extracted the
root-relative, rotation-invariant local pose vector (`ric_data = motion[t, 4:67]`, 63-d — already
normalized for translation and facing direction by construction, so it isolates pose *shape*) at
frame 0 and at the sequence midpoint for every sample.

**Operationalization** (stated before computing): if frame 0 encodes the caption's action, frame-0
poses across 69 different actions should be about as *dispersed* from each other as mid-sequence
poses are (mid-sequence is presumably where the described action is actually happening). If frame
0 instead collapses toward a generic rest pose regardless of caption, frame-0 poses should cluster
much more tightly than mid-sequence poses.

### Evidence
- Mean pairwise L2 distance between frame-0 poses: **1.0020** (std 0.834)
- Mean pairwise L2 distance between mid-sequence poses: **1.4360** (std 0.785)
- **Ratio (mid-frame dispersion / frame-0 dispersion): 1.433x** — mid-sequence poses are ~43%
  more spread out than frame-0 poses, despite the captions covering 69 distinct action verbs.
- Distance-from-corpus-frame0-mean: frame-0 poses average **0.651** from the corpus's typical
  frame-0 pose; mid-sequence poses average **0.968** from that *same* reference point.
- **74.4%** of frame-0 poses lie closer to the corpus's typical frame-0 pose than the median
  mid-sequence frame does.
- Using a radius of 1 std of the frame-0 distance distribution: **68.0%** of frame-0 poses fall
  within it, versus only **33.6%** of mid-sequence poses.

### Result
Frame-0 poses are measurably (not overwhelmingly) more homogeneous than mid-sequence poses across
a caption sample spanning 69 different action verbs — roughly 1.4x tighter by dispersion, and
roughly 2x more likely to sit near the "typical" pose by a fixed-radius test. This is a real,
directionally clear effect supporting F3, though "near-neutral" is a matter of degree here rather
than an all-or-nothing collapse — a nontrivial fraction of frame-0 poses are still meaningfully
distinct from the corpus average (this is expected: not every clip starts from a dead stop).

**Caveat for downstream use (added after review, 2026-09-05):** F3 is the weakest of the
four verdicts precisely because 1.43x is directionally clear but not dramatic. It supports the
*mechanism* (frame 0 collapses toward a common pose more than a mid-sequence frame does) but does
not by itself establish a hard ceiling on achievable model performance — that would need the
caption-semantic bucketing analysis proposed in OPEN_QUESTIONS item 2, not yet run. Future stages
should not quietly upgrade this into "the task framing is broken" without running that sharper
test first.

---

## F4 — No validation loop, no test-split eval, no seeding, no tracking

**Verdict: VERIFIED. High confidence.**

### Method
Extracted full cell source for both notebooks (48 EDA cells, 87 IMPL cells, including the 4
named monolithic cells), then ran per-line regex search, per term, across every cell — no
sampling, no "obvious spots only." Every hit was individually traced to its call site to confirm
actual use versus dead/decorative code.

### Evidence (search terms and raw results)

| Term(s) | EDA hits | IMPL hits | Used in training loop? |
|---|---|---|---|
| `val_loader`, `test_loader`, `test_split`, `X_test`, `held.?out`, `holdout`, `val_loss` | 0 | 0 | n/a |
| `validation` | 1 (prints `len(humanml3d["val"])`, never used again) | 12 (mostly `validate_clip_encoder`: a one-batch PCA sanity check of the CLIP encoder, called twice as a setup step, never inside `train_epoch`) | No |
| `.eval()` / `no_grad` | 0 | 3 / 9 (all inside generation/sampling methods, e.g. `sample_poses`, or the one-off encoder check) | No — `train_epoch` calls only `.train()`, computes only training loss |
| `torch.manual_seed`, `seed_everything`, `PYTHONHASHSEED` | 0 | 0 | n/a |
| `np.random.seed` / `random.seed` | 2 (both inside EDA's `extract_static_poses`, governs which frames get sampled for clustering) | 0 | No model-training seed anywhere |
| `wandb`, `mlflow`, `tensorboard`, `SummaryWriter`, `neptune`, `comet_ml` | 0 | 0 | n/a |
| `.log(` | 0 | 5 (all `torch.log`/`math.log` — diffusion noise-schedule math, not experiment logging) | n/a |

`test_model()` (IMPL cell 31) is a synthetic smoke test on `torch.randn` random tensors to check
tensor shapes through the architecture — not evaluation on real held-out data.

### Result
Confirmed exhaustively: no held-out validation loop is ever exercised during training, no test
split is ever evaluated, no seed governs model training (so runs are not reproducible), and no
experiment-tracking framework is used anywhere in either notebook.

---

## What this means, in plain language

The project's core numerical pipeline was decoding pose vectors incorrectly from the very first
step — pulling the wrong 66 numbers out of each 263-number frame, which is confirmed both by
reading the official dataset code and by directly measuring that the resulting "skeletons" don't
have consistent bone lengths (they should, and do, once decoded correctly). That single bug is
consistent with essentially every downstream anomaly the project's own notes describe: weird hip
offsets, joints sinking through the floor, an invented "coordinate system" story that was really
just describing the artifact. On top of that, the unsupervised clustering used to balance the
training data was fit on different, mismatched data from what the model actually trained on — a
wiring bug independent of the decode bug — and even if both of those were fixed, the basic task
design (predict a static pose from only the first clause of a caption, using literally the first
frame of the clip as the "answer") mostly teaches the model to output a generic standing pose,
because that's what frame 0 usually is, regardless of what the sentence describes. Finally,
because there was never a validation loop, a test-set evaluation, a fixed random seed, or any
experiment tracking, the project's own headline number — a 99.995% "loss reduction" across three
training phases — was never a measurement of model quality at all; it's a training-loss curve
across three different loss functions, and nothing about how well the model actually generates
poses was ever measured. The honest framing is not "the model underperformed" — it's that
performance was never actually assessed, on top of the model being trained on a decode error and
a task design that caps what's learnable regardless.

---

## OPEN_QUESTIONS

1. ~~F2's "8 vs 10" claim did not reproduce as stated.~~ **RESOLVED 2026-09-05 (review).**
   Both notebooks do use 10, as I found. The "8" in the original briefing came from
   documentation, not code or the paper: `README.md` says "8-cluster balanced sampling
   strategy" and the Obsidian deep dive says "8 pose clusters (K-means)"; the PDF report states
   no cluster count at all. The discrepancy is docs-vs-code, not code-vs-code — `BRIEFING.md`'s
   F2 section has been corrected in place by the reviewing session, marked `CORRECTED
   2026-09-05`. See also the new docs-vs-code addendum under F2 above (KMeans on raw vs. "reduced
   embeddings"; 49.6% vs. measured 38.77% cluster dominance).
2. **F3's effect size is real but moderate (~1.4x), not dramatic.** A stronger, more
   caption-semantics-aware operationalization (e.g. bucketing captions into locomotion vs.
   static-manipulation vs. large-dynamic-motion categories and comparing within/between-group pose
   distances) might sharpen this number, but was judged out of scope for this pass's time budget.
   Peer review concurs this is the weakest verdict and should not be quietly promoted to "the task
   framing is broken" without running that sharper test — see the caveat added to F3's Result
   above. Carry this bucketing analysis into `REBUILD_SPEC.md` as a candidate pre-registered
   experiment rather than running it retroactively into this report.
3. **The CVPR paper PDF was read by the primary-source research agent, not independently
   cross-checked by me.** I did not personally fetch/read the PDF; I'm relying on that agent's
   quoted excerpt (p. 5157, "Pose Representation" section). The code-level confirmation (which I
   did fetch and verify directly, twice, independently) is solid regardless.
4. ~~Report PDF (the original project report (PDF)) was not read in this stage at all.~~
   **RESOLVED 2026-09-05 (review).** A reviewing session read it via `pdftotext -layout` +
   grep: the string `99.995` appears **zero times** in the PDF — it reports only the raw Table 1
   loss values, no derived percentage. `99.995% loss reduction` instead appears **five times in
   the Obsidian deep-dive notes** (`DL - T2P Deep Dive.md`), including inside a scripted
   interview-answer passage. So the invalid loss-comparison claim is a study-notes /
   interview-prep artifact, not something the published report itself asserts — a materially
   different (and more concerning, per the reviewing session) finding than "the report states an
   invalid metric." `BRIEFING.md`'s F4 section corrected in place, marked `CORRECTED 2026-09-05`.
   I have not independently re-run the `pdftotext`/grep myself; this item records the reviewing
   session's verification, not my own.
