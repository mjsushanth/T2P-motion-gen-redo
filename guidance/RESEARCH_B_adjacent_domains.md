# Research B: Adjacent-Domain Literature Scan for Text-to-Motion (2023-2026)

Scope: NOT text-to-motion papers. Ideas from robot learning, video generation/4D, 3D human
understanding, spatial language grounding, and motion representation learning that could
transfer to generating human motion from text. Every claim below is tagged VERIFIED (I
fetched the source with WebFetch) or UNVERIFIED (surfaced by search only, abstract/claims
not independently confirmed by fetching the page). No numbers are quoted from a source I
did not fetch.

---

## 1. Robot learning / embodied AI

### Action representation is not one settled vector — it's a live design axis
Unlike HumanML3D's fixed 263-d per-frame vector (root velocity + joint positions/rotations/
velocities + foot contact, all in one flat frame-wise representation), robot learning
treats "what the action vector even is" as a first-class design decision, empirically
compared:

- **"Demystifying Action Space Design for Robotic Manipulation Policies"** (arXiv:2602.23408)
  — VERIFIED. Compares absolute vs. delta actions and joint-space vs. task-space
  (end-effector pose) parameterizations across 13,000+ real-world rollouts on a bimanual
  robot and 500+ trained models. Finding: **delta actions consistently improve
  performance**; joint-space favors control stability, task-space favors generalization —
  i.e., there is no single dominant representation, the two axes trade off different
  properties. https://arxiv.org/abs/2602.23408

- A second benchmarking paper, "Benchmarking Action Spaces in Reinforcement Learning for
  Vision-based Robotic Manipulation" (arXiv:2606.18594) — UNVERIFIED (search-surfaced only)
  — reportedly compares joint position increments (dq), joint velocity (v), pose increments
  (dx), and pose velocity (ẋ), and claims joint velocity gives the smoothest trajectories
  (lowest jerk) and best sim-to-real transfer. Treat this specific claim as unconfirmed
  until fetched directly.

**Transfer implication**: HumanML3D's representation was fixed once (Guo et al. 2022) and
essentially never re-examined as a design axis in T2M work the way robotics treats action
space. Given the direct empirical evidence above that delta/velocity-based parameterizations
reduce jerk and improve consistency, an ablation swapping HumanML3D's absolute-plus-velocity
mix for a cleaner delta-only or velocity-only parameterization is a testable, adjacent-field-
motivated experiment (see ranked list below).

### Diffusion/flow policies as the direct analog of motion diffusion models
- **Diffusion Policy** (arXiv:2303.04137, Chi et al., RSS 2023) — VERIFIED. Represents a
  robot's visuomotor policy as a conditional denoising diffusion process over an *action
  sequence* (receding-horizon control + a time-series diffusion transformer), rather than a
  single-step explicit or implicit policy. Reported average improvement of 46.9% over prior
  SOTA across 12 tasks / 4 manipulation benchmarks. This is architecturally close to motion
  diffusion models (MDM-style), but the object being diffused is a *chunk of future actions
  conditioned on current observation*, not a whole clip conditioned only on text — i.e. it
  is fundamentally a receding-horizon, closed-loop formulation. https://arxiv.org/abs/2303.04137

- **pi0: A Vision-Language-Action Flow Model for General Robot Control** (arXiv:2410.24164,
  Physical Intelligence) — VERIFIED. Fine-tunes a pretrained VLM to output continuous
  actions via **flow matching** (not autoregressive discretization), inheriting internet-
  scale semantic/visual knowledge from the VLM backbone, then specializing on ~10,000 hours
  of robot manipulation data across 7 robot configurations (68 tasks). The abstract page
  did not expose exact chunk-length/dimensionality numbers — flagged as a gap, not invented.
  https://arxiv.org/abs/2410.24164

- **FAST: Efficient Action Tokenization for VLAs** (arXiv:2501.09747, Physical Intelligence)
  — VERIFIED. The counter-approach to flow matching: compress action *sequences* via a
  discrete-cosine-transform (DCT)-based scheme (not per-timestep binning, which "performs
  poorly when learning dexterous skills from high-frequency robot data") into discrete
  tokens for autoregressive generation. pi0-FAST reportedly matches pi0-diffusion
  performance while training up to 5x faster; a universal FAST+ tokenizer was trained on 1M
  real robot trajectories. https://arxiv.org/abs/2501.09747
  **Transfer angle**: motion tokenizers in T2M (VQ-VAE-based, e.g. T2M-GPT/MoMask lineage)
  use codebook quantization; a frequency-domain (DCT) tokenizer of motion sequences is an
  unexplored, directly transferable alternative that specifically targets high-frequency /
  fine-grained motion (fingers, fast transitions) where VQ codebooks blur detail.

- **Action Chunking with Transformers / ACT** (arXiv:2304.13705, Zhao et al., "Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware") — VERIFIED. Learns a
  generative model (CVAE) over *chunks* of future actions rather than one action at a time,
  explicitly to reduce compounding error over long horizons; reports 80-90% success on six
  real tasks from only 10 minutes of demonstration data. The abstract itself did not state
  chunk length or the exact action vector format — flagged as unconfirmed rather than
  guessed. https://arxiv.org/abs/2304.13705
  **Transfer angle**: action chunking is functionally close to autoregressive motion
  generation with a fixed look-ahead window (e.g., streaming/causal motion models) — the
  CVAE-over-chunks idea is a lighter-weight alternative to full-sequence diffusion for
  low-latency, streaming text-to-motion.

- **Real-Time Chunking (RTC)** (arXiv:2506.07339) — VERIFIED. Solves the boundary-
  discontinuity problem that action-chunking policies create (jerky transitions between
  consecutive predicted chunks) via an inference-time trick: begin generating the next chunk
  while executing the current one, "freezing" already-committed actions and "inpainting" the
  rest — no retraining required, compatible with any diffusion/flow policy. Evaluated on 12
  Kinetix sim tasks and 6 real bimanual tasks (NeurIPS 2025). https://arxiv.org/abs/2506.07339
  **Transfer angle**: directly relevant to any streaming/long-horizon text-to-motion model
  that generates in fixed-length windows — the freeze-and-inpaint trick could remove the
  visible "seams" between consecutively diffused motion chunks without retraining.

### Humanoid whole-body control (motion tracking, not motion generation from text)
- **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion**
  (arXiv:2508.08241) — UNVERIFIED (search-surfaced; the search snippet reproduced what
  appears to be the actual abstract, but I did not independently fetch the page, so treat
  specifics as unconfirmed). Claimed: a guided-diffusion policy that tracks large-scale
  *kinematic* motion references (i.e., the exact kind of clip a T2M model would output) and
  reproduces them as dynamic, physically valid robot motion (jumping spins, sprints,
  cartwheels), then composes primitives zero-shot at test time via simple cost functions.
  **Transfer angle**: this is the clearest existing bridge between "a generated kinematic
  motion clip" (what T2M outputs) and "physically valid execution" — a physics-based
  post-hoc validity filter/re-targeter for T2M outputs, in the spirit of guided diffusion
  cost functions rather than full RL retraining, is worth studying directly from this paper.

- Retargeting-specific work such as **Human2Humanoid** and **OmniH2O** (search-surfaced only,
  UNVERIFIED, not independently fetched) reportedly address physics-aware cross-morphology
  retargeting from human mocap to robot morphology — conceptually the inverse operation of
  what a T2M pipeline would need if it ever had to validate generated motion against a
  physical body (self-penetration, foot-skate, joint limits, angular-momentum conservation).

---

## 2. Video generation & 4D

### Video models as emergent zero-shot vision generalists
- **"Video models are zero-shot learners and reasoners"** (arXiv:2509.20328, Wiedemer,
  Li, Vicol, Gu, Matarese, Swersky, Kim, Jaini, Geirhos; Sept 2025, DeepMind) — VERIFIED.
  Central claim: Veo 3 performs a wide range of tasks it was never explicitly trained for —
  object segmentation, edge detection, image editing, physical-property understanding,
  affordance recognition, tool-use simulation, and visual reasoning (maze solving, symmetry
  completion) — echoing the "GPT-3 moment" narrative for video, i.e. video generation models
  may be on a path toward general-purpose vision foundation models rather than narrow
  generators. https://arxiv.org/abs/2509.20328 — Project page: video-zero-shot.github.io.
  The fetched abstract did not enumerate motion-specific benchmarks/numbers — flagged as a
  gap rather than invented.
  **Transfer angle**: if a large video model already contains an implicit, zero-shot notion
  of "how a body moves under a described action" (a text->video->implicit-motion path), that
  is a candidate source of supervision or distillation signal entirely orthogonal to mocap
  datasets — see VideoMDM below for a concrete instance of this idea already executed.

### Extracting 3D motion supervision from 2D video at scale (the "sidestep small mocap" question — directly answered: yes, this is happening)
- **VideoMDM: Towards 3D Human Motion Generation From 2D Supervision** (arXiv:2606.13364)
  — VERIFIED. Trains a 3D motion diffusion prior **without any 3D ground truth**, using only
  accurate 2D keypoints extracted from monocular video. Key trick: a depth-weighted 2D
  reprojection loss is shown to be equivalent in expectation to direct 3D supervision, so
  standard 3D motion regularizers (velocity consistency, representation alignment) can be
  adapted to a 2D-supervision setting. Reported result: FID of 0.88 on HumanML3D vs. 0.54
  for the fully-3D-supervised MDM baseline — a real but non-trivial gap, and on real-world
  data (Fit3D, NBA) humans reportedly preferred VideoMDM's outputs. This is the most direct
  and recent (2026) confirmation that motion-from-video-at-scale is an active, viable
  research direction rather than a hypothetical. https://arxiv.org/abs/2606.13364

- **MotionLib: Scaling Large Motion Models with Million-Level Human Motions**
  (arXiv:2410.03311) — VERIFIED (abstract only; deeper pipeline details such as which HMR
  method was used for pseudo-labeling were reported in search snippets, e.g. "WHAM," but I
  could not confirm this from the abstract page itself — flagged UNVERIFIED for that
  specific detail). Confirmed from the fetched abstract: MotionLib claims to be "the first
  million-level dataset for motion generation," at least 15x larger than existing
  counterparts, with hierarchical text descriptions, plus a "Motionbook" tokenizer (a 2D
  lookup-free motion tokenizer designed to preserve fine-grained motion detail while
  expanding codebook capacity). No benchmark numbers were visible in the fetched abstract.
  https://arxiv.org/abs/2410.03311

- **Motion-X++: A Large-Scale Multimodal 3D Whole-body Human Motion Dataset**
  (arXiv:2501.05098) — VERIFIED. Confirmed scale: 19.5M frame-level 3D whole-body pose
  annotations, 120.5K motion sequences, 80.8K source RGB videos, 45.3K audio files, built via
  "a scalable annotation pipeline that can automatically capture 3D whole-body human motion
  and comprehensive textual labels from RGB videos" (i.e. internet video, not mocap).
  Includes hand gesture and facial expression channels absent from HumanML3D's body-only
  vector. https://arxiv.org/abs/2501.05098

- **WHAM: Reconstructing World-Grounded Humans with Accurate 3D Motion** (arXiv:2312.07531)
  — UNVERIFIED (search-surfaced only). Reportedly lifts 2D keypoint sequences to 3D using
  mocap-trained priors fused with video features, and uses SLAM-estimated camera angular
  velocity plus a contact-aware trajectory refinement to get a *global*, foot-slip-reduced
  trajectory from in-the-wild video — this is the kind of method MotionLib and similar
  pipelines reportedly build on for pseudo-labeling at scale.

- **AnthroTAP: Learning Point Tracking with Real-World Motion** (arXiv:2507.06233) —
  VERIFIED. Not directly a motion dataset, but methodologically adjacent: fits SMPL to
  humans detected in ordinary video, projects mesh vertices to get pseudo-labeled point
  tracks, resolves occlusion via ray-casting, filters unreliable tracks via optical-flow
  consistency — then trains a state-of-the-art point tracker on this pseudo-labeled data
  with only one day of training on 4 GPUs. https://arxiv.org/abs/2507.06233
  **Transfer angle**: this is a template for "human-motion-structure as a source of free,
  scalable, real-world supervision" applied to a *different* downstream task (tracking, not
  motion generation) — the pipeline design (SMPL-fit -> project -> occlusion-resolve ->
  optical-flow-consistency-filter) is a reusable recipe for building any new pseudo-labeled
  dataset from raw video cheaply.

---

## 3. 3D human understanding (HMR / SMPL pipelines feeding the above)

- Human Mesh Recovery has moved from single-image transformer regressors (HMR 2.0,
  TokenHMR — UNVERIFIED, search-surfaced) toward promptable and online video-based
  variants: **PromptHMR** (arXiv:2504.06397, UNVERIFIED) reportedly reformulates HPS
  estimation via spatial/semantic prompts; **OnlineHMR** (arXiv:2603.17355, UNVERIFIED)
  reportedly achieves constant-time per-frame inference for world-grounded motion recovery
  via a key-value cache design, relevant to any pipeline wanting to pseudo-label motion from
  video at scale without an expensive optimization pass per clip.
- **SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation**
  (arXiv:2501.09782, UNVERIFIED, search-surfaced) is reportedly a scaled-up expressive
  (body+hands+face) HMR model — relevant as the likely annotation engine behind Motion-X++
  style datasets, though this specific link was not confirmed by fetching either paper.
- **Net effect for T2M**: the HMR/SMPL pipeline is now mature and fast enough (per-frame,
  cached inference; promptable; expressive whole-body) that "distill motion from arbitrary
  YouTube video of a described action" is an engineering-feasible data-augmentation strategy
  today, not a research risk — Motion-X++ and MotionLib are existence proofs, VideoMDM is
  the strongest direct evidence it also improves an actual generative motion model.

---

## 4. Language grounding for space and body

### CLIP is measurably bad at exactly the kind of language T2M prompts need (left/right, behind, raise, bend)
- **"What's left can't be right — The remaining positional incompetence of contrastive
  vision-language models"** (arXiv:2311.11477) — VERIFIED. Confirms CLIP-style models
  systematically fail at left-right positional relations, and that this is *predictable*
  from properties of the pretraining data (not a fixable-by-scale problem): captions
  containing both objects in both left/right orientations essentially never co-occur in the
  same training minibatch, so the contrastive objective never gets the signal it needs to
  separate "A left of B" from "B left of A." **Fix demonstrated**: synthetic data with
  controlled left-right pairs, which transfers to real images with measurable improvement on
  Visual Genome Relations. Exact accuracy numbers were not visible on the fetched abstract
  page (flagged, not invented). https://arxiv.org/abs/2311.11477

- **"What's 'up' with vision-language models? Investigating their struggle with spatial
  reasoning"** (ACL/EMNLP 2023, aclanthology.org/2023.emnlp-main.568) — VERIFIED. Built the
  **"What'sUp" benchmark**: photo sets that vary *only* in spatial relation between two
  objects, identity held fixed. Evaluated 18 VL models; all underperformed badly relative to
  humans. Confirmed specific number: **BLIP fine-tuned on VQAv2 scored 56% vs. 99% for
  humans** on this benchmark, despite BLIP being near-human on VQAv2 itself — i.e. general
  VQA competence does not imply spatial-relation competence. Root cause confirmed from the
  abstract: large pretraining corpora (LAION-2B) contain too little reliable spatial-relation
  language, and simple fixes (up-weighting prepositions, fine-tuning) did not resolve it.
  https://aclanthology.org/2023.emnlp-main.568/

- **LRR-Bench: Left, Right or Rotate? Vision-Language Models Still Struggle With Spatial
  Understanding Tasks** (arXiv:2507.20174) — VERIFIED. Tests absolute 2D spatial position
  (left/right in an image) and 3D spatial understanding (movement, rotation). Confirmed
  finding: VLMs reach human-level only on the two simplest task variants; on harder
  (rotation/3D) tasks, "the best-performing VLMs even achieve near-zero scores." Exact
  numeric accuracy table was not visible in the fetched abstract (flagged, not invented).
  https://arxiv.org/abs/2507.20174

**Why this matters directly for T2M**: HumanML3D-style pipelines and most T2M models use a
frozen CLIP text encoder (following Guo et al.'s evaluator design) to embed prompts before
motion generation/scoring. The three papers above are independent, converging evidence
(2023-2026) that this exact embedding space is specifically bad at directional/spatial/
relational language — precisely the vocabulary that distinguishes "raise your left arm" from
"raise your right arm," or "step behind" from "step in front of." This is a **directly
falsifiable, laptop-testable hypothesis** for the T2M project (see ranked list).

### What's replacing CLIP for spatial grounding (3D-specific literature, mostly scene-level not body-level)
- Search-surfaced only (UNVERIFIED, not fetched): approaches augmenting visual/language
  tokens with explicit 3D positional embeddings derived from depth/camera parameters
  (LLaVA-3D-style), or explicit pairwise spatial-relation encodings added on top of/instead
  of a similarity-only contrastive space (e.g. arXiv:2603.24721 "Scalable Object Relation
  Encoding for Better 3D Spatial Reasoning in LLMs", arXiv:2605.30307 "Grounded 3D-Aware
  Spatial Vision-Language Modeling"). None of these are body-pose specific — they target
  3D scene/object grounding, not human articulation — so the transfer is at the level of
  *technique* (explicit relational/positional structure added on top of a similarity
  embedding) rather than a drop-in replacement. No claim here is treated as confirmed since
  neither paper was fetched.

---

## 5. Motion representation learning (beyond the standard Guo et al. evaluator)

- **TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis**
  (arXiv:2305.00976, Petrovich et al., ICCV 2023) — VERIFIED. Directly targets the
  "standard Guo et al. evaluator" the prompt asks about: extends TEMOS with an added
  contrastive (InfoNCE-style) loss on top of the synthesis objective, on KIT-ML and
  HumanML3D. Confirmed concrete result: **median retrieval rank improved from 54 to 19**
  versus the prior evaluator-style approach — a substantial, directly quoted improvement.
  Also introduces a formal text-to-motion retrieval benchmark protocol (as opposed to only
  generation-quality metrics). Exact R@1/R@5 numbers were not visible on the fetched
  abstract page (flagged, not invented). https://arxiv.org/abs/2305.00976
  Follow-on work (search-surfaced only, UNVERIFIED): "Beyond Global Alignment: Fine-Grained
  Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning" (arXiv:2601.21904) and
  "Multi-Modal Motion Retrieval by Learning a Fine-Grained Joint Embedding Space"
  (arXiv:2507.23188) reportedly push toward *fine-grained* (sub-sequence / body-part level)
  motion-language alignment rather than TMR's single global embedding per clip/sentence —
  directly relevant if the T2M project's failure mode is compositional (e.g. gets the action
  right but not the body part or direction).

- **Motion foundation models** are an emerging but thin category as of this scan:
  "MoFM: A Large-Scale Human Motion Foundation Model" (arXiv:2502.05432, UNVERIFIED,
  search-surfaced) reportedly targets one-shot/unsupervised/supervised downstream transfer
  (action classification, anomaly detection) rather than generation. "UMO: Unified
  In-Context Learning Unlocks Motion Foundation Model Priors" (arXiv:2603.15975,
  UNVERIFIED) is a 2026 paper explicitly framing motion priors as something to be unlocked
  via in-context learning rather than task-specific finetuning. Neither was fetched; treat
  as leads, not confirmed findings.

- **Masked motion/video modeling as self-supervised pretraining** — UNVERIFIED, search-
  surfaced only, not fetched. Multiple motion-aware masked-autoencoder variants exist in the
  video-understanding literature (e.g. "Self-supervised Video Representation Learning with
  Motion-Aware Masked Autoencoders," arXiv:2210.04154; "Masked Motion Encoding for
  Self-Supervised Video Representation Learning," arXiv:2210.06096) which predict masked
  patches *and* motion structure jointly, rather than pixels alone. These target video
  action-recognition pretraining, not 3D motion generation — the transfer would be: pretrain
  a motion-sequence encoder via masked-frame/masked-joint reconstruction (an unsupervised
  objective needing no text at all) before ever touching the small labeled HumanML3D corpus,
  analogous to how masked video pretraining bootstraps action recognition from unlabeled
  video.

---

## Ranked: transferable ideas a solo researcher could test on a laptop in under 10 GPU-hours

Ranked by (evidence strength) x (expected information gained) / (implementation cost). All
are scoped to fit a single consumer GPU or a small cloud instance within a 10 GPU-hour
budget — i.e., small-scale ablations/probes, not full retraining runs.

1. **Probe the CLIP text encoder actually used in your T2M pipeline for the exact spatial
   failure documented above.** Take the frozen CLIP encoder your model conditions on, embed
   ~200-500 minimal-pair prompts differing only in a spatial/directional term ("raise your
   left arm" vs "raise your right arm"; "step behind the box" vs "step in front of the
   box"; "bend forward" vs "bend backward" — modeled directly on the "What'sUp" minimal-pair
   design, arXiv 2023.emnlp-main.568, and the left-right probe in arXiv:2311.11477), and
   measure embedding cosine similarity/separability (e.g. a simple linear probe or nearest-
   neighbor classification on the embeddings). Cost: <1 GPU-hour (no training, just
   inference + a scikit-learn probe). This directly tests, on your own model's failure
   surface, whether the well-documented CLIP spatial blind spot (VERIFIED above) is present
   in your specific pipeline before you invest in a fix.

2. **Swap in a contrastive text-motion encoder (TMR-style) as an auxiliary loss or evaluator,
   instead of relying solely on the Guo et al. evaluator's embedding space.** TMR's
   improvement (median rank 54->19, VERIFIED, arXiv:2305.00976) came from adding an InfoNCE
   contrastive loss on top of an existing synthesis objective — the code is public
   (github.com/Mathux/TMR). Fine-tuning or applying a pretrained TMR text/motion encoder as
   an added retrieval-style evaluation metric (or a fine-tuning signal) on a subset of
   HumanML3D is a few hours on one GPU, and gives you a second, independently-validated axis
   of "does the text and motion actually match" beyond FID/R-precision from the standard
   evaluator.

3. **Delta/velocity-only ablation of the motion representation, motivated directly by the
   robotics finding that delta actions consistently outperform absolute ones**
   (arXiv:2602.23408, VERIFIED, 13k+ rollouts / 500+ models). HumanML3D's 263-d vector
   already mixes absolute root position with local velocities; a clean ablation — train (or
   fine-tune) a small motion VAE/diffusion model on a stripped-down, purely delta/velocity
   representation vs. the standard mixed representation, on a small subset of HumanML3D, and
   compare foot-skate / jerk / FID — is a direct, falsifiable transplant of a result from an
   adjacent field. Feasible in under 10 GPU-hours if restricted to a small model and subset
   of the data (e.g. a few thousand clips, short training run for a relative-comparison
   signal rather than SOTA numbers).

4. **Frequency-domain (DCT) motion tokenizer as an alternative to VQ-VAE codebook
   quantization, motivated by FAST** (arXiv:2501.09747, VERIFIED — DCT-based tokenization
   outperforms per-timestep binning specifically for high-frequency, fine-grained action
   data). Implement a DCT-based tokenizer for a HumanML3D motion clip (this is pure signal
   processing, no training required beyond quantizing DCT coefficients) and compare
   reconstruction fidelity against your existing VQ tokenizer, specifically on fast/fine
   sub-motions (finger movement proxies, quick direction changes) where VQ codebooks are
   known to blur detail. Near-zero GPU cost (mostly CPU-side signal processing); useful as a
   feasibility probe before committing to retraining a full tokenizer.

5. **Freeze-and-inpaint chunk stitching (RTC-style, arXiv:2506.07339, VERIFIED) for any
   autoregressive/streaming motion generation you already have**, to remove visible seams
   between generated windows — this is an inference-time algorithm requiring **no
   retraining**, so it is nearly free to test (just inference-time compute) if your
   pipeline already generates motion in fixed-length chunks; if it doesn't, this idea is
   not applicable and should be skipped rather than forced.

6. **(Higher cost, still plausible in-budget) Small-scale 2D-reprojection auxiliary loss,
   VideoMDM-style** (arXiv:2606.13364, VERIFIED — FID 0.88 vs. 0.54 fully-3D-supervised, so
   a real but bounded gap): take a handful of easily obtainable YouTube clips of a single
   describable action, run an off-the-shelf 2D pose estimator, and add a depth-weighted 2D
   reprojection auxiliary loss during fine-tuning of an existing small motion model, to
   directly test whether *any* signal from cheap 2D video improves generation on an
   out-of-HumanML3D-distribution action category. This is the most ambitious item on this
   list to fit in 10 GPU-hours — scope it to one action category and a short fine-tune run,
   not a full retrain, and treat a null result as informative rather than a failure.

Items 1 and 2 are the highest-confidence, lowest-cost starting points: both rest on
VERIFIED, fetched, quoted results, both require no new training data, and both can be run
in under an hour of actual GPU time with the rest of the 10-hour budget held in reserve for
whichever of items 3-6 the first two results motivate.
