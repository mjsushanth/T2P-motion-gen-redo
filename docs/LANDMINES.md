# LANDMINES — traps that produce plausible wrong answers with no error raised

> **Read this before writing any code in this repository.**
>
> Every item here produces output that looks entirely reasonable and is wrong. None of them
> raises an exception. That is what makes them worth a dedicated document: a crash tells you
> it is broken, and these do not.
>
> Items 1-6 were found by reading the original project. Items 7-10 are general to this
> domain and this hardware. **Add to this file every time you find a new one.**
>
> Format: **the trap** -> **what it looks like** -> **the evidence** -> **what to do instead**.

---

## 1. `motion[:66]` is not 22 joints of XYZ

**Status: VERIFIED (2026-09-05). Confirmed against `../primary_source/motion_representation.ipynb`
and empirically by the bone-length invariant test — see the evidence block below and
`../FORENSICS.md`.**

**The trap.** HumanML3D motion vectors are `[T, 263]`. It is natural to assume the first 66
values are 22 joints x 3 coordinates. The canonical layout (Guo et al., CVPR 2022,
`motion_representation.py`) is:

```
index      contents                                          size
--------   -----------------------------------------------   ----
[0]        root angular velocity (about Y)                      1
[1:3]      root linear velocity (x, z)                          2
[3]        root height (y)                                      1
[4:67]     ric_data - local positions, 21 NON-root joints      63
[67:193]   rot_data - 6D continuous rotations, 21 joints      126
[193:259]  local velocities, 22 joints                         66
[259:263]  foot contact flags                                    4
                                                        total = 263
```

So `[:66]` is **4 root scalars plus 62 of the 63 position values**. Reshaped to `(22, 3)`,
"joint 0" is `[root_rot_vel, root_vel_x, root_vel_z]` — a velocity triple, not a position —
and every joint after it is **shifted by 4 scalars**, splicing X/Y/Z across joint boundaries.

**What it looks like.** Everything works. The array reshapes cleanly. The skeleton plots.
It looks vaguely humanoid but subtly wrong, and you spend three months inventing explanations
for the wrongness: a mysterious coordinate convention, unstable bone lengths that need a
reference database and post-hoc clamping, a degenerate Z axis, left-right confusion.

**The evidence.** Two independent lines, both VERIFIED here.

*Primary source.* `primary_source/motion_representation.ipynb`, the official HumanML3D
processing code, builds the vector as:

```python
data = root_data                                        #   4
data = np.concatenate([data, ric_data[:-1]], axis=-1)   #  63
data = np.concatenate([data, rot_data[:-1]], axis=-1)   # 126
data = np.concatenate([data, local_vel],     axis=-1)   #  66
data = np.concatenate([data, feet_l, feet_r], axis=-1)  #   4
```

*The invariant test.* Bone lengths, coefficient of variation across samples
(`primary_source/f1_bone_length_results.npz`, 21 bones):

| decode | bone-length CV |
|---|---|
| correct `ric` decode (`[4:67]`) | **0.000 on all 21 bones** (std exactly 0) |
| original's `motion[:66]` reshaped `(22,3)` | **0.068 to 0.817**, median ~0.16 |

The original code path is `HumanML3DProcessor.extract_pose` in `DL_T2P_IMPL.ipynb`:
`motion_sequence[frame_idx][:66]` on a `[T, 263]` array. The original EDA notebook had
independently hypothesised `66 pos + 66 vel + 126 rot + 5 = 263` from correlation heatmaps.

**The corollary nobody noticed.** The correct decode's bone-length std is *exactly zero* —
HumanML3D retargets every subject onto **one canonical skeleton**. Bone lengths are a
**constant of the dataset**, not a quantity to be learned or penalised. The original project's
entire anatomy apparatus — the reference-bone-length database, the weighted bone-length loss,
the multi-stage FK clamping, the "Phase 2 breakthrough" — was machinery built to recover a
constant that the data hands you for free. See §8.

**Do instead.** Use `recover_from_ric(motion, 22)` from the official HumanML3D repo, or slice
`[4:67]` and prepend an explicit zero root. Never index a packed representation by guess.

**Generalisation:** this applies to any packed motion/pose format — SMPL parameter vectors,
AMASS, MotionX. Get the layout from the code that *wrote* it, not from a heatmap.

---

## 2. Two different wrong layouts can sum to the same total

**The trap.** `4 + 63 + 126 + 66 + 4 = 263`. Also `66 + 66 + 126 + 5 = 263`. The second is
wrong. The arithmetic agreeing is what let the error survive an entire project.

**What it looks like.** You sanity-check your understanding of a format by adding up the
component sizes, the total matches the documented dimension, and you move on satisfied.

**Do instead.** Validate a layout hypothesis with an **invariant**, never with a sum. For
poses the invariant is free: **bone lengths are constant for a given subject.** Decode, compute
the 21 bone lengths, and look at the coefficient of variation across frames and samples.
Correct decode -> CV under a few percent. Wrong decode -> large. This test takes ten minutes
and would have saved the original project.

**Generalisation:** every packed format has an invariant. Find it before you trust a slice.

---

## 3. The caption and the frame may describe different things

**The trap.** HumanML3D captions describe a *motion sequence*. Taking frame 0 as "the pose
for this caption" pairs an action description with whatever the actor happened to be doing
before they started.

**What it looks like.** Frame 0 of *"a person walks forward"* is a neutral standing pose.
Frame 0 of *"a person does a cartwheel"* is **also** a neutral standing pose. A large fraction
of training pairs therefore teach `(action text -> rest pose)`. The model dutifully learns
that mapping, the loss goes down, and generation is semantically random. No error is raised
because nothing is broken — the data genuinely says that.

**Do instead.** Either select a representative frame (peak deviation from the sequence mean,
or a pose-change-rate criterion), or use a dataset built for static poses in the first place
(PoseScript). Whatever you choose, **measure the label noise first and report it**: sample N
pairs and quantify how often the frame is near-neutral regardless of the verb.

---

## 4. A falling training loss is not a result

**The trap.** Reporting training loss as evidence of quality, and worse, comparing training
losses across configurations whose loss functions differ.

**What it looks like.** "Final loss 1.52e15 -> 1.17 -> 0.69, a 99.995% reduction." Three
numbers, three different objectives, the first a bug artifact. It reads like a result table.
It is not one. It cannot be — a loss is only comparable to another loss over the same
objective on the same data.

**The evidence.** Table 1 of the original `the original project report (PDF)`.

**Do instead.** Nothing is a result until it comes off the evaluation harness on a held-out
split, and the harness itself has been validated against a published number. Loss curves go
in a diagnostics figure, never in a results table. If you catch yourself writing a percentage
reduction in loss, delete it.

---

## 5. Cluster labels fitted on one thing, applied to another

**The trap.** The original clustered poses taken from a **random** frame per sequence, on the
**raw un-normalised** `[:66]` slice, then used those cluster IDs to select training samples
whose pose is the **first** frame.

**What it looks like.** A principled-sounding "cluster-aware sampling strategy" that corrects
a 49.6% dominant-cluster bias. Plots look fine. Distributions rebalance. But the labels
describe different vectors than the ones being sampled, and because the slice included root
height and root velocity, K-means was partly clustering *how fast the root was moving*.

**The evidence.** `extract_static_poses` in the original EDA notebook uses
`frame_idx = random.randint(0, len(motion)-1)`; the training processor uses
`frame_selection='first'`. The EDA `main()` also uses `n_clusters=10` while the paper and the
sampling config say 8.

**Do instead.** Fit and apply any grouping on **exactly** the vectors you will train on, after
the same normalisation. Assert it: the array you cluster and the array you sample must be the
same object, and a test should enforce that.

---

## 6. CLIP cannot reliably tell left from right

**The trap.** Using CLIP's global text embedding as the sole conditioning signal for
spatially-specific language.

**What it looks like.** The model generates a good pose with the wrong arm, consistently,
and you conclude the architecture needs more capacity.

**The evidence.** The original measured cosine distance 0.03 between the embeddings of
"person raising left arm" and "person raising right arm" — they are nearly the same vector.
UNVERIFIED here; re-measure before relying on the figure.

**Do instead.** Treat laterality as a known weakness and design around it: token-level
conditioning rather than a single pooled vector, a text encoder with better compositional
behaviour, or explicit mirror augmentation (flip the pose, swap left/right in the caption).
Whatever you pick, **report a laterality-specific metric** so the failure is visible.

---

## 7. MPS is not deterministic

**The trap.** Setting a seed on Apple Silicon and expecting reproducible numbers.

**What it looks like.** You run the same config twice, get 0.41 and 0.44, and attribute the
difference to a change you made in between.

**Do instead.** Run every comparison with multiple seeds and report the spread. If the
seed-to-seed spread is a large fraction of the effect you are claiming, you have not measured
an effect. Say so.

---

## 8. Anatomy enforced by loss is anatomy you do not have

**The trap.** Predicting joint positions and adding a bone-length penalty to make them
plausible. The penalty fights the data term, needs a weight you must tune, and still lets
invalid poses through.

**Do instead.** Parameterise in **rotation space** — joint rotations plus a fixed skeleton —
and decode with a differentiable forward-kinematics layer. Bone lengths are then correct by
construction, exactly, with no loss term, no reference database and no post-hoc clamping.
This is what MDM and MotionDiffuse do. If you find yourself building a bone-length loss,
stop and ask whether you have chosen the wrong output space.

---

## 9. Post-hoc "constraint enforcement" hides model quality

**The trap.** Clamping generated poses to valid bone lengths after sampling, then evaluating
the clamped output.

**What it looks like.** Every generated pose is anatomically perfect. You have measured your
clamp, not your model.

**Do instead.** Always report metrics **both** pre- and post-correction, and treat the gap as
a diagnostic of how much the model is not doing. A large gap is a finding, not a detail.

---

## 10. A metric implementation you did not validate is not a metric

**The trap.** Implementing FID / R-Precision / MM-Dist from a paper description and reporting
the output. These have many silent failure modes: wrong feature extractor, wrong pooling,
biased covariance estimate, batch-size dependence, a retrieval pool that differs from the
standard protocol.

**What it looks like.** Plausible numbers in the right ballpark that are not comparable to
anything published.

**Do instead.** Before your harness is allowed to produce a headline number, **reproduce a
published figure with it** to a stated tolerance. If you cannot, say exactly why and label
every subsequent number as internally-comparable-only. This is Stage 3's gate and it is the
single most important thing this project does differently from its predecessor.
