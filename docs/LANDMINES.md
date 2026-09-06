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

**The evidence.** Table 1 of the original the original project report (PDF) reports the three raw
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

**Correction (2026-09-05, review):** an earlier version of this entry said "the paper and
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

**Status: VERIFIED (2026-09-05, review audit of `DL_T2P_IMPL.ipynb` cell 47).
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

**It was not an oversight — it was the documented design.** (Corroboration found by the audit
of this entry, 2026-09-06; verified independently by the author before adding.) Cell 46's own
markdown, verbatim:

> "implementation uses progressive guidance scaling **during training** (2.0->7.0 over 50, 100
> epochs)."
>
> "we use `self._forward_with_text` twice, once with `null_text_emb` and once with `text_emb`,
> helping us on CFG formula. similarly, **loss uses** `run two parallel forward` concept, two
> predictions are co[mbined]"

This raises the severity rather than lowering it. A typo gets caught by the next reader; a
believed-correct design gets *written up as a contribution* and defended. The original report
lists "Dual-Path Classifier-Free Guidance" among Phase 3's seven headline improvements. Nobody
was going to find this by re-reading the code, because the code matched the intent exactly —
the intent was wrong. **The only thing that catches this class of error is an evaluation that
would have shown the text conditioning doing nothing.** Which is the whole argument for D-02.

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

---

## 13. R-Precision is meaningless without its candidate-pool size

**Status: VERIFIED in this repository, 2026-09-06, by our own code making the mistake (E0 v1).**

**The trap.** R-Precision is retrieval accuracy over a pool of one correct caption plus N-1
distractors. **The number is only interpretable against a stated N**, and the field's HumanML3D
protocol fixes N = 32. Build the pool from the whole batch instead and you get a valid-looking
accuracy that is not comparable to anything published.

**What it looks like.** E0 v1 pooled 200 candidates and reported R-Precision-top3 = **0.280**.
Nothing errors. The number is a correct measurement of a different quantity. Re-batched to 32,
the same data and the same checkpoint gave **0.710**. A 2.5x swing from a batching detail.

**Do instead.** Record `r_precision_batch_size` in the record JSON — E0's does — and state N
beside every R-Precision figure. Chance is `k/N`, so quote that too: top-3 of 32 is ~9.4% chance,
which is what makes 0.72 meaningful.

**Extended, 2026-09-06 (Stage 5 demo work, review SUP-20260906-58/62/63): the pool size is not a
configuration detail — it is the range the instrument is calibrated over at all, and outside
that range the instrument does not degrade gracefully, it stops meaning anything.** Built a
retriever using the same validated text-motion embedding space (`EvaluatorMDMWrapper`'s text
encoder) this project's R-Precision numbers already come from, intending to use it for full-
corpus nearest-neighbour retrieval (~8,198 real motions, not a 32-candidate batch). Measured
before shipping it (not assumed to work because the underlying component was well-validated
elsewhere): full-corpus self-retrieval — does a caption's own true motion rank first among all
8,198 candidates — collapsed to **~1%**. Diagnosed directly, not just observed: a real caption's
own true motion scores **0.978** cosine similarity against its own text embedding, yet ranks
**264th out of 8,198**, because every corpus motion occupies a tight **0.97-0.99** cosine band at
this scale. A TF-IDF retriever on the identical sample found the true motion first **74.7-78.3%**
of the time (two independent measurements, `demo/measure_self_retrieval.py` and a separate
director-run measurement) — proving the collapse is specific to this embedding space at this
scale, not a property of the retrieval task itself.

**The reason, stated plainly: the text encoder was trained to discriminate a caption from ~31
distractors (R-Precision's own batch-of-32 protocol), never to rank it first against thousands of
candidates.** This is a non-obvious property of an instrument the whole text-to-motion field
shares, not a one-off implementation bug — it explains *why* the published protocol specifies
N=32 rather than that being an arbitrary convention: at this embedding space's actual resolution,
batch-of-32 is roughly the largest pool size the instrument still functions over at all.
**R-Precision is not a general-purpose retrieval system that happens to be evaluated on batches
of 32 for convenience — outside that pool size, in this same embedding space, the statistic is
not just less precise, it is not measuring retrieval ability in any recoverable sense.**

**Do instead, generalized beyond R-Precision specifically:** before repurposing a metric's
underlying embedding space for a *different* task at a *different* scale than the one it was
validated on (here: full-corpus retrieval, versus 32-candidate batch retrieval), measure the new
task directly rather than assuming validation at one scale transfers to another. A component
being well-validated for its original, narrower purpose is not evidence it generalizes to a
broader one — check by looking at real outputs (a caption's own true-motion rank, not just
whether the code runs) before trusting it.

---

## 14. FID is badly biased when n is not much larger than the feature dimension

**Status: VERIFIED in this repository, 2026-09-06 (E0 progression, real-vs-real).**

**The trap.** FID estimates a 512x512 covariance from n samples. When n is close to d the
estimate is severely under-conditioned and FID is biased **upward**, so a perfectly good model
looks bad — and the bias shrinks as n grows, which makes the metric look like it is "improving"
when only the sample count changed.

**The evidence, from this project's own runs** (identical data, checkpoint and code; only n and
the R-Precision batching changed):

| n per subset | n/d (d=512) | real-vs-real FID |
|---|---|---|
| 200 | 0.4 | 0.745 |
| 1024 | 2.0 | 0.173 |
| 2099 | 4.1 | **0.029** |

Real data against real data should score ~0. It scores 0.745 at n=200. **The entire signal there
is estimator bias.** Note the v2->v3 drop is *super-linear* — 2x the samples, 6x lower FID —
which is the signature of leaving the badly-conditioned regime, not of ordinary 1/n convergence.

**The diagnostic consequence, which is the reason this entry exists.** R-Precision does **not**
estimate a covariance; it is batch-wise retrieval over a fixed 32-candidate pool, so it is
insensitive to total n. In the same progression it moved 0.710 -> 0.720 while FID moved 6x.
**So when FID and R-Precision both sit below a published reference, they usually do not have the
same cause, and one explanation cannot cover both.** Sweep n: whatever moves is estimator bias,
whatever does not move is a real difference in protocol, checkpoint, or model.

**Do instead.** Record `fid_embedding_dim` and n in every record JSON (E0's does both). Never
compare FID across runs with different n. Get n comfortably above d — the published protocol's
`--repeat_time` averaging is partly what buys this. And before attributing an FID gap to your
model, sweep n and confirm the gap survives.

**A second, sharper illustration: three FID values from one bit-identical generated set.**
The real-vs-real sweep above varies both sides of the comparison. This one holds the *generated*
side completely fixed — the same 128 cached motions, scored three separate times, only the
reference computation varying — which isolates the estimator problem from any question about
whether generation itself is noisy:

| round | what varied | FID |
|---|---|---|
| 1 | first ground-truth reference draw | 1.0731 |
| 2 | ground-truth reference redrawn (generation held fixed by MDM's default deterministic seed) | 1.3997 |
| 3 | fixed, full-scale (n=4640) ground-truth reference — same cached generated motions, zero regeneration | 3.2909 |

Same 128 generated motions, three different numbers, a 3x spread top-to-bottom. Fixing the
reference's own sample size (round 3) did not shrink the gap — it grew, because the *generated*
side is still n=128 and a 512-dim covariance from 128 samples is rank-deficient (rank <=127)
regardless of how well-estimated the other side of the comparison is. **The lesson is stronger
than "get n big on both sides": a well-conditioned reference paired against a rank-deficient test
covariance is still an unusable comparison.** Full detail in `docs/EXPERIMENT_LOG.md`'s E0b entry
and `LEDGER.md` Item 16; the regime-dependent consequence for which metric gates E1/E2 is in
`docs/DECISIONS.md` D-25.

---

## 15. A multi-line packed field parsed with a single split silently drops every sample after the first

**Status: VERIFIED (2026-09-06), found while building E0b (`scripts/materialize_humanml3d_test_subset.py`).**

**The trap.** The HF `TeoGchx/HumanML3D` dataset's `caption` field looks like one string, but it
is actually **multiple newline-separated entries**, each independently in the official
`"<caption>#<tokens>#<from_tag>#<to_tag>"` format — HumanML3D's own convention for a sequence
with several human-written annotations. A single `caption_field.split("#")` on the whole
multi-line string looks completely reasonable and reshapes cleanly into `[caption, tokens,
from_tag, to_tag]` — the first four `#`-delimited pieces are exactly what you'd expect. What it
actually produces for `to_tag` is the true first-entry value with the **entire second caption
entry silently concatenated onto it** (`"0.0\nperson walking at a average pace..."`), and every
entry after the second is dropped from the parse entirely.

**What it looks like.** No exception at parse time — the corrupted line is written to disk and
looks like a slightly odd two-line text file. The actual failure surfaces one layer downstream
and looks like nothing happened at all: MDM's own `Text2MotionDatasetV2.__init__` (and this
project's own vendored copy of the same class) wraps each sample's per-line processing in a bare
`try/except: pass`. The malformed second line raises (`list index out of range` trying to read a
`tokens` field from a line with no `#` in it), the exception is swallowed, and that sample is
silently dropped from `data_dict`/`name_list` — **not once, but for every sample in the
materialized set**, because the bug is systemic across all rows, not sample-specific. The
downstream symptom was `real_num_batches: 0` and an empty generated dataset — no traceback, no
warning, just a dataset that loads successfully and iterates zero times.

**The evidence.** Found by directly inspecting `repr(ex['caption'])` on a real HF row rather than
reasoning about the format from a truncated print statement (Stage 1's own earlier caption
inspection had only ever printed `str(...)[:200]`, which happened to cut off before the second
caption entry began, and that partial view was carried forward as an assumption without being
re-checked here). The raw field, verbatim: `'a person is walking in place at a slow pace.#a/DET
person/NOUN is/AUX walk/VERB in/ADP place/NOUN at/ADP a/DET slow/ADJ pace/NOUN#0.0#0.0\nperson
walking at a average pace forward, swaying arms and torso with a sense of swagger#person/NOUN
walk/VERB at/ADP a/DET average/ADJ pace/NOUN forward/ADV sway/VERB arm/NOUN and/CCONJ
torso/VERB with/ADP a/DET sense/NOUN of/ADP swagger/NOUN#0.0#0.0\n...'` — three complete,
correctly-formatted entries, joined by newlines, that a whole-field `.split("#")` treats as one.

**Do instead.** Split on newlines *first* to recover the individual entries, then treat each one
independently — never assume a packed field is single-valued just because a naive parse of it
produces a plausible-looking, correctly-shaped result. As with §1/§2: validate a parsing
hypothesis against an invariant (here, "does the resulting dataset have the sample count I
expect," not "did the split produce four pieces without erroring") rather than trusting that the
arithmetic came out looking right.

**Generalisation:** any format library that swallows per-record parse errors (a `try/except:
pass` around per-sample processing, common in dataset-loading code written to tolerate a few bad
files) will silently absorb a systemic parsing bug as if it were normal missing-data filtering.
Check the *count* of what survived a bulk load, not just that the load completed without error.

---

## 16. Naming a limitation in a pre-registration is not the same as checking whether it disables the check

**Status: VERIFIED in this repository, 2026-09-06 — this is a review-discipline trap, not a
domain trap like the entries above, and it belongs here anyway because this project produced a
documented instance of it (review SUP-20260906-37, on `docs/EXPERIMENT_LOG.md`'s E1A-power
pre-registration).**

**The trap.** The E1A power check's pre-registration stated, in its own "Does NOT establish"
line, that it trained and evaluated on the same materialized subset, and characterized this as
showing the architecture "*can* exploit text at this budget when the eval captions were also
seen in training, which is a **necessary but weaker condition** than generalizing to held-out
captions." That sentence is honest, specific, and was written down *before* the run — every
surface signal of rigor is present. **It is also wrong**, because "weaker" implies the same kind
of evidence at reduced strength, when the actual failure mode is categorical: a model can score
above the pre-registered chance threshold (3/32 = 0.09375) purely by memorising which of ~4,648
training captions pairs with which training motion, with zero generalisable text conditioning
learned. That is not a weaker version of the signal the gate exists to detect — it is a different
mechanism that produces the identical observable number. **The gate could have passed for a
reason unrelated to the question it was built to answer**, and the disclosure of the limitation
would have made that failure *harder* to notice on a later re-read, not easier, because the
caveat reads like the risk was already considered and accepted.

**Why this is worse than an undisclosed limitation.** An undisclosed train/test overlap looks
unfinished and invites scrutiny. A disclosed one, phrased as "a real limitation, stated here, not
hidden," looks like due diligence already performed — it manufactures exactly the false comfort
that rigor is supposed to prevent. The tell was available at write-time: the disclosure named the
condition ("evaluated on seen captions") but never asked what a model doing *only* that, and
nothing else, would score on the gate's own metric. Working that out is one more step past
naming the limitation, and it is the step that actually matters.

**Do instead.** For any pre-registered check with a stated limitation, ask explicitly: "under
this limitation alone, with none of the capability the check is trying to detect, what would the
gate's own metric read?" If that hypothetical failure mode can clear the pre-registered threshold,
the limitation does not weaken the check — it invalidates it, and the check needs a design fix
(here: disjoint train/eval splits, materialized before running), not a footnote.

**Generalisation:** this applies to any gate, not just ML power checks — a benchmark run on
warmed cache, a security test against a non-production config, a load test with the rate limiter
disabled. In each case, ask whether the stated caveat is a *dial* (the same signal, turned down)
or a *different mechanism* that can independently satisfy the pass condition. Only the first kind
is safely absorbed by a caveat; the second kind requires redesigning the check.

---

## 17. A stochastic arm's across-seed spread is treatment variance, not measurement noise

**Status: VERIFIED in this repository, 2026-09-06 (review SUP-20260906-47, E1-pilot's
random-window control) — the second time this exact pattern has surfaced (E0a flagged the same
shape as its own likely explanation via MDM's `repeat_time` averaging, never fully confirmed
there; this entry is the confirmed instance).**

**The trap.** Comparing a deterministic arm (e.g. a fixed truncation rule) against a stochastic
arm (e.g. a randomly-placed window) by drawing the stochastic arm **once** and reseeding only the
retrieval-batch shuffle looks like an apples-to-apples comparison — both arms get "a seed," both
get re-measured. It is not apples-to-apples. Reseeding the deterministic arm changes only which
samples land in which batch (measurement noise). Reseeding the stochastic arm changes the
*treatment itself* — a different random window is a different intervention, not a different
sample of the same one. A single draw's across-seed swing is therefore dominated by
treatment variance, and "the two arms scored about the same" from one draw cannot distinguish
"the manipulation has no effect" from "this particular draw happened to land close to the other
arm."

**The evidence, from this project's own run:** one random-window placement draw scored
R-Precision-top3 = 0.5339, close enough to the prefix control (0.5456) to read as "no detectable
difference." Averaging 8 independent placement draws (same keys, same batch-shuffle seed held
fixed so only placement varied) gave a mean of **0.5279** with a real spread across draws
(std 0.0082, range 0.5132-0.5384) — the single draw had simply landed on the higher end of that
range. The averaged mean vs. the prefix control (gap 0.0177, ~6x the averaged mean's own standard
error) revealed a real effect the single draw's "no difference" reading had obscured.

**Do instead.** Before comparing a stochastic arm to a deterministic one, ask which seed is doing
which job. If reseeding changes the treatment (not just the batch order, sample order, or other
incidental randomness), that arm needs its own averaging loop — treat it the way `repeat_time` /
`mm_num_repeats`-style parameters already treat MDM's own stochastic generation — before its
score is compared to anything deterministic. A single draw of a stochastic treatment is a sample
size of one no matter how large the underlying dataset is.

**Generalisation:** this applies anywhere a comparison mixes a fixed condition against a
randomized one — A/B tests where "B" is itself a random policy, ablations where one arm samples
a hyperparameter, benchmarks where one competitor is stochastic and the other isn't. The fixed
condition's seed controls incidental variance; the randomized condition's seed controls the
treatment. Averaging the wrong one, or averaging neither, produces a comparison that looks
symmetric and isn't.

---

## 18. A hypothesis and a success criterion are not a pre-registration without a power calculation

**Status: VERIFIED in this repository, 2026-09-06 (review SUP-20260906-55, E1B) — a
review-discipline trap, alongside §16 and §17, not a domain one.**

**The trap.** E1B was pre-registered in the sense this project had been using the word all day:
a stated hypothesis ("E1B worse than E1A") and a stated success criterion ("by more than the
seed-to-seed spread"), both written down in `REBUILD_SPEC.md` §6 before the run. That looked
complete. It was not: nobody asked, before spending ~5 hours of CPU time across training and
generation for both arms, **what sample size would be needed to detect the effect actually
expected, and whether this run's planned n reached it.**

**The check that would have caught it, computable in about a minute from numbers already on
hand:** at R-Precision ~0.30 with n=128 generated samples per arm, the standard error of an
A-vs-B gap is ~0.057, so the minimum effect detectable at 3σ is ~0.17. The best available prior
estimate of the true effect — this project's own same-day E1-pilot measurement — was 0.145
(corpus-wide, unattenuated) to 0.27 (conditional), and every stated reason available at the time
said the generation-side effect should be **attenuated relative to the retrieval-space number**,
not equal to it. So the honest pre-run statement, achievable before training started, was: *"n=128
resolves this only if the effect survives into generation completely undiminished, which is not
expected."* The run went ahead anyway, and the resulting gap (0.047, at 0.80σ) confirmed the
underpowered case exactly.

**Do instead.** Treat pre-registration as three parts, not two: hypothesis, success criterion,
and a minimum-detectable-effect calculation against the planned sample size, using whatever prior
estimate of the effect is available (even a labelled guess is better than none). State the
verdict — powered or underpowered — before running, and if underpowered, state explicitly why the
run is worth doing anyway (pipeline validation, a null that usefully bounds something, trivial
marginal cost). What is not acceptable is running an underpowered comparison without knowing it
is underpowered, because then a noise result gets read as a finding — which is the specific
failure this whole project exists to guard against, arrived at by a different route.

**Generalisation:** any A/B comparison, ablation, or benchmark with a fixed, costly sample size
should have this calculation done before the run starts, not reconstructed afterward to explain
a surprising result. A hypothesis plus a criterion answers "what would convince us." A power
calculation answers "can this run possibly convince us at all" — and only the second question
determines whether spending the compute is worthwhile in the first place.

---

## 19. A correct computation can still produce a display that supports the opposite conclusion

**Status: VERIFIED in this repository, 2026-09-06 (Stage 5 demo work, review SUP-20260906-68) —
the fourth review-discipline entry in this file, alongside §16-18, and like them more
transferable than most of the domain findings above.**

**The trap.** Two real bugs were caught in this project's own interactive demonstrator, in the
same session, both by the same method: running the tool with a fresh, uncurated, real input
rather than reading its code. In both cases, **the underlying computation was correct.** Only
what the interface chose to display, and how a viewer would read it, was wrong.

**Instance one:** an embedding-based retriever was measured before shipping (per §18's own
discipline) and found to collapse at full-corpus scale — caught by looking at real query
outputs. That catch was itself about a computation being *wrong* at that scale, not this entry's
concern. **Instance two, this entry's actual example:** a TF-IDF retrieval panel correctly
computed and displayed cosine similarity for two queries of different length (a full caption and
its truncated form) side by side. TF-IDF cosine similarity is not comparable across queries of
different length — a shorter query mechanically scores higher, having fewer terms left unmatched
in its own vector, independent of whether it found the right answer. On a real, uncurated test
caption, the truncated query's displayed score was *higher* than the full query's — a viewer
reading the two numbers side by side would conclude "truncation improved the match," the exact
opposite of the page's entire argument. **The similarity computation was correct. The bug was
showing two individually-correct, differently-scaled numbers as if they meant the same thing.**

**Why static review does not catch this.** Reading `retrieval.py`'s cosine-similarity code shows
correct math. Reading `app.py`'s display code shows two numbers being printed, formatted
correctly, no type error, no exception. **Nothing about the code, read in isolation, signals
that the two numbers are being placed in a context where a reader will compare them to each
other** — that only becomes visible when a real pair of differently-shaped queries produces two
numbers next to each other and a human looks at what they imply together.

**Do instead.** Before shipping any interface that displays two or more numbers side by side
inviting comparison, ask specifically: *is this quantity actually comparable across the contexts
being juxtaposed*, not just *is each number computed correctly on its own*. If the answer is no
(different query lengths, different sample sizes, different reference distributions), either
remove the numbers (if the qualitative content — which answer was returned — already carries the
point, as it did here) or state the incomparability explicitly next to the numbers, not in a
separate caveat section a reader may not reach. And run the interface with real, varied,
uncurated input before trusting that a static read of the code caught everything — static review
checks the computation; only exercising the thing with real input checks what the page as a
whole invites a reader to conclude.

**Generalisation:** this applies to any interface, report, or dashboard displaying more than one
number of the same apparent type — accuracy scores at different sample sizes, latencies under
different loads, similarity scores for queries of different length or specificity. A number can
be individually correct and still be dangerous once placed next to another number a reader will
naturally compare it to.

---

## 20. "I did not find X" is only "X does not exist" if the search was exhaustive

**Status: VERIFIED in this repository, 2026-09-06 (`RESULTS.md`'s F1-F8 miss, review
SUP-20260906-69) — the fifth review-discipline entry, alongside §16-19.**

**The trap.** Writing `RESULTS.md`, a claim was needed about how many findings this project's
forensics stage produced. `FORENSICS.md` was checked (Stage 1's own empirical write-up: F1-F4)
and `LEDGER.md`'s audit of a later set of findings was checked (F6-F8). Neither mentioned an F5.
**Concluded: F5 was never assigned.** It was wrong — F5 (the original project's engineering
state: monolithic notebook cells, triplicated classes, no seeds, hardcoded paths) was defined in
`BRIEFING.md`, verified by direct code inspection during the original task specification, and was
never in either of the two places actually searched.

**Why this is a distinct failure from reusing a wrong number (§16, §18-19 catalogue variations on
that instead).** A *presence* claim ("X is true") comes with a citation by construction — you
quote or point at the thing that makes it true, and a reader can check that one source. An
*absence* claim ("X does not exist," "X was never done," "no evidence of X") has no such natural
anchor. It is only as strong as the search that produced it, and **the search's own scope is
usually invisible in the final sentence** — "F5 was never assigned" reads identically whether it
followed an exhaustive search of every relevant document or a check of the two most convenient
ones.

**Do instead.** Before writing any claim of the shape "X does not exist" / "was never done" /
"no such thing was found," name the search scope explicitly, even just to yourself: which
documents, which code paths, which time range did the search actually cover, and is that
scope *complete* for the kind of thing being claimed absent — or just the most obvious or
recently-handled subset. If the true scope needed for confidence is larger than what was
actually checked, say "not found in X and Y" rather than "does not exist," and treat the
stronger claim as unearned until the wider search actually happens.

**Generalisation:** this applies to any absence claim in any domain — "no relevant prior work,"
"this bug was never reported," "no test covers this case," "nothing else depends on this
function." Each is only as strong as an explicit, checkable search scope, which is far easier to
state honestly before writing the conclusion than to reconstruct after someone else finds the
counterexample.

---

**Second instance (2026-09-06), and this one cost two sessions.** The Stage 5 demo's live
generation was recorded in the worker's ledger as "hung (near-0% CPU) for over 35 minutes after
finishing its first sampling loop." The reviewing session then built its own detector to reproduce
the hang — `find /var/folders -maxdepth 4 ... -name "*.mp4"` — watched CPU fall below 5% for 90
seconds with no file found, and duly reported `HANG REPRODUCED`.

Nothing was hung. Both generations had completed and both `.mp4` files had been written, valid and
decodable. The real path was `.../t2p_demo_gen_*/full/samples_00_to_00.mp4` — **one directory
deeper than `-maxdepth 4` reaches.** The process was idle because it was finished.

Two independent sessions concluded "hung" from an absence of output that was really an absence of
looking, and the second one *confirmed* the first by repeating its mistake in a different form.
A reproduction is only corroboration if the two attempts could have failed independently; here both
inferred a negative from a search neither had validated could find a positive. **Before trusting a
detector that reports absence, feed it a case you know is present.** Had the detector been run once
against the already-completed output sitting on disk from the earlier invocation, it would have
reported nothing then too — and the flaw would have surfaced in seconds.


## 21. A redaction cannot be documented by quotation

**Status: VERIFIED in this repository, 2026-09-06 (the git-history identifier scrub, review
SUP-20260906-50) — the sixth review-discipline entry, and the only one about how the record
itself works rather than about how claims are checked.**

**The trap.** Joel authorized scrubbing six literal identifying strings (a course code, an
institution name, a group number, a compound archive path) out of this repository's git history.
Documenting *that* the scrub happened is normal, expected practice — except a document describing
a literal-string removal necessarily needs to name what was removed, and naming it means quoting
it. **A ledger entry that quotes the six strings while describing their removal reintroduces
every one of them into the current, public HEAD of the very repository the scrub was meant to
clean** — committed normally (not as part of the history rewrite itself), so never touched by
the rewrite, sitting in plain text on GitHub the moment it is pushed.

**This happened twice, independently, within about an hour, which is the strongest evidence
behind any of the six lessons in this section.** First in this project's own `LEDGER.md`: an
entry documenting the scrub's replacement rules quoted the literal mapping verbatim. Caught,
fixed. Then the exact same pattern turned up in the peer reviewing session's own finding about
that first mistake — the review *warning that you cannot document a literal-string scrub using
the literals* itself quoted the literals, in the same paragraph, into the same public repository.
Two independent authors, warned by the same near-miss, made the identical mistake immediately
after describing it.

**Why this forces a real exception to append-and-annotate.** Every other correction in this
project preserves the original wrong text, marked superseded, because the original's wrongness is
part of the record worth keeping. **This class is different: the original text's wrongness *is*
the leaked data.** Preserving it to show what was corrected would mean preserving exactly the
information the correction exists to remove. When the content itself is the defect, annotation
cannot fix it — only replacement can. This is a real, load-bearing exception to a standing rule,
not a convenience.

**Do instead.** Describe the *shape* of what was redacted (a course code, a compound path, a
group number) without reproducing it. Keep the literal mapping in a gitignored file, not a
committed one — the same pattern already used for this project's own `.archive_path`. If a
correction to a redaction-related document is ever needed, edit in place and say so explicitly,
rather than following the append-and-annotate convention that governs every other correction in
this project.

**Generalisation:** applies to any documentation of a redaction, takedown, or PII/secret removal
in any project — a changelog entry, a security advisory, an incident postmortem. The instinct to
show exactly what was removed, for auditability, is correct everywhere else in a project's
record and is precisely backwards here.

---

## Review-discipline lessons (§16-21), gathered

**Neither collaborator on this project was reliably right; the practice of re-verifying rather
than trusting was.** Across one day of work between two independent sessions, each caught real
errors in the other's output and, on inspection, in its own — and in more than one case a
correction to a correction is what produced the number that finally held up. Every entry below
was caught the same way: not by either party being careful in general, but by neither party
accepting a claim, a correction, or an absence of a hit as settled until it was re-derived from
source.

Six entries in this file are not about this project's own domain (motion generation, diffusion
models, evaluation metrics) but about the process of producing and reviewing research work
itself. Gathered here as one list because they transfer further than anything else in this
document — they would apply to a different project in a different field unchanged. Cross-
referenced from `RESULTS.md`.

- **§16 — Naming a limitation in a pre-registration is not the same as checking whether it
  disables the check.** A disclosed limitation can manufacture false comfort precisely because
  disclosure looks like rigor.
- **§17 — A stochastic arm's across-seed spread is treatment variance, not measurement noise.**
  Reseeding a deterministic arm controls incidental variance; reseeding a stochastic one changes
  the treatment itself. Averaging the wrong one produces a comparison that looks symmetric and
  isn't.
- **§18 — A hypothesis and a success criterion are not a pre-registration without a power
  calculation.** "What would convince us" and "can this run possibly convince us at all" are
  different questions, and only the second determines whether the run is worth its cost.
- **§19 — A correct computation can still produce a display that supports the opposite
  conclusion.** Static review checks the computation; only running the thing with real, varied,
  uncurated input checks what the interface as a whole invites a reader to conclude.
- **§20 — "I did not find X" is only "X does not exist" if the search was exhaustive.** An
  absence claim's strength is bounded by its search scope, which is usually invisible in the
  sentence stating the conclusion.
- **§21 — A redaction cannot be documented by quotation.** Describe the shape of what was
  removed, keep the literals outside version control, and accept that this one class of
  correction must overwrite rather than annotate — the one lesson here about how the record
  itself works, rather than how a claim gets checked.

Each was found in this project by the same underlying practice: re-deriving a claim (one's own,
or a peer's) from source before accepting it, rather than trusting that a plausible-looking
result, an absent hit, or a clean run meant the work was done.


---

## 22. A reproduced error is a fact about a configuration, not a property of the hardware

**How it presented here.** Training on `--device mps` raised
`TypeError: Cannot convert a MPS Tensor to float64 dtype`. This was reproduced deliberately rather
than assumed, reported rather than silently worked around, and written up as D-24: *"Training is
CPU-only. MPS is unusable for this codebase."* Every wall-clock estimate in the project was then
computed from the CPU rate, and D-26 stopped an experiment partly because those estimates made it
unaffordable.

The reproduction was sound. The **generalisation from it was not.** "This code path fails on MPS
as vendored" was written down as "MPS is unusable" — and then read by everything downstream as
"this machine cannot do GPU training."

**The tell that it was a generalisation, not a finding.** D-24 contained its own reversal clause,
naming the fix and calling it *"a real option and it is not large."* The author of the constraint
already knew it was probably removable and recorded that in the same breath as the constraint. The
clause then sat untested for eight hours. **A reversal condition you can state precisely and
cheaply test is not a caveat — it is an unrun experiment.** If you can name the test, the honest
options are to run it or to say plainly that you chose not to and why.

**What it actually cost.** The fix was one line, and provably bit-identical rather than merely
close: `_extract_into_tensor` did `.to(device)[t].float()` — ship float64 to the device, then cast.
Reordering to `.float().to(device)[t]` casts first. Indexing is a pure gather, so cast-then-gather
and gather-then-cast return the same bits, and the original threw the float64 away on the very next
operation anyway. Measured result: 2.284 s/step CPU to 0.231 s/step MPS, **9.95x**, on the same
machine and seed. Roughly a hundred hours of projected compute was a hundred hours because of the
order of two method calls.

**The structural version.** This is §4's error (a measurement mistaken for a conclusion) relocated
from a metric to the infrastructure — and infrastructure is where it is most dangerous, because a
metric gets re-examined when it looks wrong, while a hardware constraint gets quietly treated as
the shape of the world and is never revisited. Nobody re-derives the floor they are standing on.

**Practice.** When an environment failure is about to become a project constraint, separate two
claims explicitly and in writing: *this configuration fails* (a measurement) and *this capability
is unavailable* (a conclusion that needs its own evidence). Before the second one is allowed to
change scope, budget, or a stopping rule, cost the smallest patch that would test it. Here that was
about fifteen minutes against a hundred projected hours.

**And it was caught from outside.** No internal review pass surfaced this — the finding came from
the author asking the obvious naive question ("MPS doesn't run here? what's going on?") that the
people deep in the work had stopped asking, because for them it had already been settled. Settled
is precisely the state in which a premise stops being examined.

---

## 23. A reviewer's assurance that a check is redundant is a claim, not a clearance

**How it presented here.** After the MPS float32 patch landed, the reviewing session wrote in
SUP-20260906-79 that the end-to-end validation run the worker had queued was *"a weaker test than
the E0a check you already passed — a full training run confounds device difference with seed
variance... The E0a check already did that job properly."* The reasoning was sound on its face:
E0a had passed on MPS with a device delta of 4e-8 against a seed delta of 2.7e-3, and a single
training run genuinely cannot separate device from seed.

**The worker ran it anyway, and it caught a second MPS bug.** `evaluator_wrapper.py`'s
`get_co_embeddings` / `get_motion_embeddings` carried the *same* cast-after-transfer defect as
`_extract_into_tensor`. The E0a gate had not covered it, because **the E0a check and the E1
pipeline instantiate two different vendored evaluator classes.** Passing one class's MPS gate
licensed nothing about the other's. Had the worker deferred, the bug would have surfaced later as
corrupted numbers in an experiment rather than as a crash in a validation run.

There was a third gap in the same area that neither the finding nor the decision record mentioned:
`dist_util.dev()` only ever returned `cuda` or `cpu`. Without an MPS branch, nothing could reach
the patched diffusion code at all. **The prescription named one fix; three were needed.** A patch
verified correct in isolation is not a patch verified sufficient in situ.

**Why the reviewer's error was structurally predictable.** SUP-79 generalised from *one* validated
path to a *different* path on the strength of the first one's clean result — which is landmine §22
("a reproduced error is a fact about a configuration, not a property of the hardware") with its
sign flipped. §22 is over-generalising a failure; this is over-generalising a success. Both come
from treating a measured fact about one configuration as a property of the system. The clean E0a
numbers made the reviewer *more* confident, not less, that further checking was waste — and
confidence is precisely what removes the impulse to check.

**Practice, for both roles.** For the producer: a reviewer's "this is redundant" carries no more
authority than any other unverified claim, and costs nothing to disregard when the check is cheap
and already queued. Run it and report the result — if the reviewer was right you have lost
minutes; if wrong you have caught something no one was looking for. For the reviewer: **downgrade
a test only when you can name what covers the gap it leaves.** SUP-79 asserted E0a "did that job
properly" without checking whether the two paths shared an evaluator class. Naming the covering
test is the discipline that would have exposed the gap before the advice was given, and it is the
same discipline §16 demands of a blocker claim.

**Cheap tests do not need a justification to run. They need one to skip.**
