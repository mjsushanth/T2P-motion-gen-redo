# GLOSSARY

> Every acronym and domain term used anywhere in this repository, expanded and defined in one
> or two sentences. **Maintained continuously — the first time a term appears in a notebook or
> document, it gets defined there AND mirrored here.**
>
> Marked **[unverified]** where the definition is from general knowledge rather than checked
> against a primary source in this repo.

---

## Datasets and formats

**AMASS** — Archive of Motion Capture as Surface Shapes. A large collection of motion-capture
datasets unified into a common SMPL-based representation. The source that HumanML3D and
PoseScript are both derived from. [unverified]

**HumanML3D** — a text-to-motion benchmark: ~23k motion sequences from AMASS, each paired with
several human-written English captions. Motions are stored as `[T, 263]` float vectors. The
standard benchmark for text-to-motion. See `LANDMINES.md` §1 for the layout. [unverified]

**PoseScript** — a dataset of *static* 3D human poses (from AMASS) paired with textual
descriptions, both automatically generated and human-written. Built for the text-to-pose task
specifically, unlike HumanML3D. [unverified — confirm size, licence and benchmark protocol]

**PoseFix** — a companion dataset for text-based *pose editing*: pairs of poses plus a text
instruction describing the modification. [unverified]

**SMPL / SMPL-X** — Skinned Multi-Person Linear model. A parametric human body model: a small
vector of shape and pose parameters maps, through a learned function, to a full 3D body mesh.
SMPL-X adds hands and face. Licence-gated; registration required. [unverified]

**ric_data** — "rotation-invariant coordinates" data. In HumanML3D's packed vector, the
root-relative 3D positions of the 21 non-root joints (63 values). Decoded with
`recover_from_ric`. [unverified]

**rot6d / 6D rotation** — representing a 3D rotation with 6 numbers (two columns of the rotation
matrix, re-orthogonalised) instead of 3 Euler angles or 4 quaternion components. Preferred in
neural networks because it is continuous, unlike Euler angles and quaternions.

**263-dimensional vector** — HumanML3D's per-frame motion encoding.
`4 root + 63 ric + 126 rot6d + 66 local velocity + 4 foot contact`. See `LANDMINES.md` §1.

---

## Task and modelling

**Text-to-pose (T2P)** — generate a single static 3D human pose from a natural-language
description. The task of this project.

**Text-to-motion (T2M)** — generate a *sequence* of poses from text. The larger, better-studied
task; HumanML3D is its benchmark.

**Diffusion model** — a generative model trained to reverse a gradual noising process. Start
from pure noise and denoise step by step, guided at each step by a condition (here, text),
until a sample emerges.

**DDPM** — Denoising Diffusion Probabilistic Model. The standard formulation: a fixed forward
noising schedule and a learned reverse denoiser trained to predict the added noise.

**CFG — classifier-free guidance** — sample the model twice per step, once with the text
condition and once with a null condition, then extrapolate:
`eps = eps_uncond + w * (eps_cond - eps_uncond)`. Larger `w` means stronger adherence to the
text and usually lower diversity.

**Cross-attention** — the mechanism that lets pose features "look at" text features: queries
come from the pose, keys and values from the text embedding.

**Forward kinematics (FK)** — computing joint *positions* from joint *rotations* plus a fixed
skeleton, by walking the kinematic tree from the root outward. Differentiable, so it can sit
inside a network. See `LANDMINES.md` §8.

**Kinematic chain / tree** — the parent-child structure of a skeleton (pelvis -> spine -> neck
-> head; pelvis -> hip -> knee -> ankle -> toe). Determines how FK propagates.

**CLIP** — Contrastive Language-Image Pre-training. A model trained to embed images and text
into a shared space. Its text encoder is often reused as a general-purpose text conditioner.
Known to handle spatial and left/right language poorly — `LANDMINES.md` §6.

**ControlNet** — an adapter that conditions an image-diffusion model on a spatial input such as
a pose skeleton. The reason text-to-pose has downstream product value. [unverified]

---

## Evaluation

> **None of these have been computed in this repository yet.** Definitions are for orientation;
> the exact protocol must come from the reference implementation, not from this file.

**R-Precision (top-1/2/3)** — retrieval accuracy: given a generated motion and a pool of
candidate captions (the true one plus distractors), how often does the true caption rank in the
top k by embedding distance. Measures text-motion alignment. [unverified protocol]

**FID — Fréchet Inception Distance** — distance between the feature distributions of generated
and real samples, computed with a task-specific feature extractor (not Inception, despite the
name, in the motion literature). Lower is better. Sensitive to sample count and to which
extractor is used. [unverified protocol]

**MM-Dist — multimodal distance** — average distance between a generated sample's features and
its conditioning text's features. Lower is better. [unverified protocol]

**Diversity** — average pairwise distance between randomly sampled generated outputs. Should be
close to the real data's value; both much lower and much higher are bad.

**MultiModality** — average pairwise distance between multiple outputs generated from the *same*
text. Measures whether the model represents a distribution or collapses to one answer.

**MPJPE — Mean Per Joint Position Error** — mean Euclidean distance between predicted and
ground-truth joint positions, in millimetres. **PA-MPJPE** first removes global rotation, scale
and translation with Procrustes alignment.

**Mode collapse** — the model produces near-identical outputs regardless of input noise.
Detected by low MultiModality.

---

## Engineering

**MPS — Metal Performance Shaders** — PyTorch's Apple Silicon GPU backend. The compute target
here. Non-deterministic; see `LANDMINES.md` §7.

**Coefficient of variation (CV)** — standard deviation divided by mean. Used here as the
bone-length invariant test: a correct pose decode gives a low CV per bone across samples.

**VERIFIED / UNVERIFIED** — this project's epistemic labels. See `00_START_HERE.md` §6.
