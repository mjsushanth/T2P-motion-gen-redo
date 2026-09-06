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

**The evidence.** Table 1 of the original `the original project report (PDF)` reports the three raw
loss values (1.52e15, 1.17, 0.69) with no derived percentage. **Correction (2026-09-05, peer
review):** the "99.995% loss reduction" phrasing itself does not appear in the PDF at all — a
`pdftotext -layout` + grep found the string `99.995` zero times in the report. It appears five
times in the Obsidian deep-dive notes (`DL - T2P Deep Dive.md`), including inside a scripted
60-second interview-answer passage. So the invalid comparison is a study-notes / interview-prep
artifact that the published report itself does not make — worth knowing precisely because it
means the number is something Joel has been rehearsing to say out loud, not just something once
written down.

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
`frame_selection='first'`. The array-position-to-dataset-row mapping is also broken: the true
row ID per clustered-array position lives only in `caption_indices.npy`, which the training
notebook never loads — empirically, `caption_indices[i] != i` for 23,382/23,384 positions
(99.99%), so training selects essentially arbitrary rows, not merely mislabelled ones
(`FORENSICS.md` F2).

**Correction (2026-09-05, peer review):** an earlier version of this entry said "the paper and
the sampling config say 8" clusters vs. EDA's 10. That was wrong. Both notebooks, as read, use
10 consistently — the code never uses 8 anywhere. The "8" traces to **documentation only**:
`README.md` ("8-cluster balanced sampling strategy") and the Obsidian deep-dive ("8 pose
clusters"); the PDF report states no cluster count at all. So the discrepancy is docs-vs-code,
not code-vs-code — a fifth instance of this project's documentation describing a cleaner
pipeline than the code implements. Two more of the same kind, also peer-review-verified: the PDF
claims K-means ran "on the reduced embeddings" (PCA/t-SNE) but the code fits it on the raw 66-d
slice; `README.md` claims sampling "avoided 49.6% cluster dominance" but the largest cluster
measured directly from `clusters.npy` is 38.77% — 49.6% does not reproduce from the saved
artifacts.

**Do instead.** Fit and apply any grouping on **exactly** the vectors you will train on, after
the same normalisation. Assert it: the array you cluster and the array you sample must be the
same object, and a test should enforce that. Also: treat this project's own docs/README/notes as
**unverified** narrative, same as any other secondary source — verify claims against code and
saved artifacts, not against what the writeup says the code does.

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
If you find yourself building a bone-length loss, stop and ask whether you have chosen the
wrong output space.

**Correction (2026-09-06, Stage 2 landscape research):** this entry previously claimed "this is
what MDM and MotionDiffuse do." Checked directly against both papers — **that is not accurate.**
MDM's own paper states its pose representation "is a sequences of human poses represented by
either joint rotations or positions... MDM can accept motion represented by either locations,
rotations, or both," and for its HumanML3D experiments specifically it uses "the same
representation" as Guo et al.'s own redundant vector — i.e. joint positions, velocities, *and*
rotations all together, predicted directly, the same representation the original failed project
misdecoded (just correctly used here). MotionDiffuse's paper is explicit that its pose state
"generally contains joint rotation, joint position, joint velocity, and foot contact conditions"
and is "robust to the various motion representations" — again the redundant vector, not a
rotation-only parameterisation. **Neither flagship model's published HumanML3D numbers rely on
predicting rotations-only-plus-FK.** The rotation-space argument above is sound **as an
engineering argument on its own terms** (HumanML3D's bone lengths are a verified dataset
constant, `LANDMINES.md` §1, so a parameterisation that makes that constancy structural rather
than learned is a real advantage) — but it should be argued from that first-principles logic, not
from a false claim about field precedent. See `REBUILD_SPEC.md` D-11 for how this is actually
argued.

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

---

## 11. Classifier-free guidance applied to the TRAINING objective

**Status: VERIFIED (2026-09-05, Opus 5 supervisor audit of `DL_T2P_IMPL.ipynb` cell 47).
This is a root-cause-grade defect on par with §1.**

**The trap.** CFG is an *inference-time* extrapolation. At training time the only correct
mechanism is **conditioning dropout**: replace the text embedding with a null embedding some
fraction of the time (~10%) so the network learns both `eps(x,t,c)` and `eps(x,t,null)`. The
original instead ran the guidance formula *inside the loss*:

```python
if text_embeddings is not None and guidance_scale > 1.0:
    null_embeddings   = torch.zeros_like(text_embeddings)
    uncond_noise_pred = self.model(noisy_poses, t, null_embeddings)
    cond_noise_pred   = self.model(noisy_poses, t, text_embeddings)
    predicted_noise   = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
...
diffusion_loss = F.mse_loss(predicted_noise, noise)
```

`self.guidance_scale` defaults to 3.0 and `train_model` ramps it 2.0 -> 7.0, so
`guidance_scale > 1.0` is **always true** during Phase 3. This branch always ran.

**Why it is fatal, not merely wrong.** Write `u = eps_uncond`, `c = eps_cond`, `w = guidance`.
The objective is:

```
    minimise  MSE( u + w*(c - u),  eps )
```

Set `c = u = eps`. Then `u + w*(c-u) = u = eps` and **the loss is exactly zero** — for any `w`.
That solution requires the conditional and unconditional predictions to be *identical*, which
is to say it requires **no text dependence whatsoever**.

**The training objective is fully satisfiable by a model that ignores the conditioning
signal.** There is no gradient pressure to use the text. This is not a subtle inefficiency; it
removes the very thing Phase 3 was built to add.

**What it looks like.** Loss falls smoothly. Phase 3's loss is the lowest of the three phases
(0.69), which reads as success. Generated poses are anatomically fine and semantically random.
The team concludes CLIP is weak at spatial language and starts tuning guidance scales. Every
downstream observation is consistent with a working system that just needs more capacity.

**Three compounding consequences:**

1. **No conditioning dropout exists anywhere in the notebook.** Both branches run on every
   sample and both receive gradient. The null branch never learns the marginal distribution;
   it learns whatever makes the *combination* fit. Note the deep-dive troubleshooting guide
   says "Ensure 10% null conditioning during training" — the notes describe the correct
   mechanism, the code does something else. See §12.
2. **The objective function changes every epoch.** Because `w` ramps 2.0 -> 7.0 across training,
   the quantity being minimised is literally a different function each epoch. Loss values are
   therefore not comparable *even within a single phase*. This compounds §4.
3. **Self-inflicted gradient instability.** At `w = 7` the two forward passes of a
   shared-weight network carry coefficients `+7` and `-6`, largely cancelling. The
   `5*tanh(x/5)` output squash and the gradient clipping that the project treats as
   architectural insights are compensations for an amplification the objective created.

**Do instead.** Conditioning dropout at training:
```python
mask = torch.rand(batch_size, device=dev) < 0.1
cond = torch.where(mask[:, None], null_embedding, text_embeddings)
loss = F.mse_loss(self.model(noisy, t, cond), noise)      # ONE forward pass
```
and apply the guidance formula **only in the sampling loop**. Prefer a *learned* null embedding
over `zeros_like`. If you ever see a guidance scale in a training signature, that is the smell.

---

## 12. The same batch's statistics used to normalise the diffusion target

**Status: VERIFIED (same audit). Two separate defects in one function.**

**12a — per-batch normalisation of `x_0`.**
```python
batch_mean = batch.mean(dim=0, keepdim=True)
batch_std  = batch.std(dim=0, keepdim=True) + 1e-5
normalized_batch = (batch - batch_mean) / batch_std
```
The clean pose is renormalised **against the current batch** before noise is added. A diffusion
noise schedule assumes a *fixed* data scale; here `x_0`'s distribution shifts every step with
whatever 96 samples happened to be drawn. Two identical poses in different batches become
different training targets. Dataset-level statistics computed once are the fix.

**12b — one timestep applied to the whole batch.**
```python
estimated_clean_pose = self.noise_scheduler.scheduler.step(
    model_output=predicted_noise,
    timestep=t[0].item(),        # <-- t is a per-sample random vector
    sample=noisy_poses
).prev_sample
```
`t` is sampled per-sample, then **only element 0's timestep** is used to step the entire batch.
Every sample except index 0 is denoised with the wrong noise level. That corrupted
`estimated_clean_pose` is precisely the input to the anatomy loss — so the "Phase 2
breakthrough" anatomy term was computed on a mis-stepped estimate for ~95 of every 96 samples.

Combined with §1 (bone lengths are a dataset constant, so the anatomy loss was recovering a
known quantity) the anatomy machinery was measuring a corrupted estimate of a constant.

**Do instead.** Normalise with fixed dataset statistics. Pass the full per-sample `t` vector to
any scheduler call. Assert it: `assert timestep.shape[0] == sample.shape[0]`.

