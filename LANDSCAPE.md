# LANDSCAPE — text-to-pose / text-to-motion, 2024-2026

Stage 2, Part A. Every claim below was fetched from a primary source (paper page, official
repo, licence page, or dataset card) by one of three parallel research passes, or is explicitly
marked UNVERIFIED where it was not. URLs are inline. This document argues nothing — it is the
evidence `REBUILD_SPEC.md` and `POSITIONING.md` argue from.

---

## 1. Text-to-motion on HumanML3D — the established benchmark

### 1.1 The evaluation protocol, in plain terms

- **R-Precision (top-1/2/3)** — given one generated motion and a pool of 32 candidate text
  descriptions (1 correct, 31 mismatched), retrieval accuracy of finding the correct text in the
  top-k of a ranked list, using a learned joint text-motion embedding space. Measures whether the
  motion actually matches its text.
- **FID (Fréchet distance)** — distance between the feature distribution of generated motions and
  real motions, in the same learned embedding space. Measures overall realism; ~0 for real data.
- **MM-Dist** — average distance between a motion's embedding and its paired text's embedding.
  Finer-grained semantic-match signal than R-Precision.
- **Diversity** — variance across embeddings of motions generated from *different* text prompts.
  Detects mode collapse across the dataset.
- **MultiModality (MModality)** — variance across *multiple generations from the same prompt*.
  Distinct from Diversity: measures whether one prompt admits multiple plausible outputs.

### 1.2 Which evaluator code everyone actually uses

**VERIFIED.** Every paper below evaluates against the same shared instrument: a contrastive
text-motion embedding model (trained via `train_tex_mot_match.py`) from Chuan Guo et al.'s own
release, official repo **`EricGuo5513/text-to-motion`**
([github.com/EricGuo5513/text-to-motion](https://github.com/EricGuo5513/text-to-motion)).
`final_evaluation.py` is the entry point; checkpoints ship under `checkpoints/t2m/text_mot_match/`.
Confirmed independently via T2M-GPT's and StableMoFusion's own READMEs, both of which explicitly
reuse this exact evaluator/checkpoint set. **This is the instrument to reproduce for D-03's
harness-validation gate** — it is a known, fetchable, third-party artifact, not something to
reimplement from a paper description (`LANDMINES.md` §10 exists precisely for this reason).

### 1.3 Published HumanML3D numbers, quoted from each paper's own table

All rows are HumanML3D **test set**, generation-given-ground-truth-text setting. "Real" is the
paper's own ground-truth reference row (FID is measured against this).

| Method | Venue | Code | R-Prec-top3 | FID | MM-Dist | Diversity | MModality |
|---|---|---|---|---|---|---|---|
| *Real* | — | — | 0.797±.002 | 0.002±.000 | 2.974±.008 | — | — |
| MDM | ICLR 2023 (Oral) | [motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model) | 0.611±.007 | 0.544±.044 | 5.566±.027 | 9.559±.086 | 2.799±.072 |
| MotionDiffuse | TPAMI 2024 | [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse) | 0.782±.001 | 0.681±.001 | 3.113±.001 | 9.410±.049 | 1.553±.042 |
| MLD | CVPR 2023 | [motion-latent-diffusion](https://github.com/ChenFengYe/motion-latent-diffusion) | 0.772±.002 | 0.473±.013 | 3.196±.010 | 9.724±.082 | 2.413±.079 |
| T2M-GPT | CVPR 2023 | [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) | 0.775±.002 | 0.141±.005 | 3.121±.009 | 9.722±.082 | 1.831±.048 |
| MoMask | CVPR 2024 | [momask-codes](https://github.com/EricGuo5513/momask-codes) | 0.807±.002 | 0.045±.002 | 2.958±.008 | — | 1.241±.040 |
| StableMoFusion | ACM MM 2024 (Oral) | [StableMoFusion](https://github.com/h-y1heng/StableMoFusion) | 0.841±.002 | 0.098±.003 | *not reported* | 9.748±.092 | 1.774±.051 |
| MotionLCM (1-step) | ECCV 2024 | [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM) | 0.803±.002 | 0.467±.012 | 3.022±.009 | 9.631±.066 | 2.172±.082 |

Sources (ar5iv HTML mirrors of arXiv, since direct PDF fetch failed for tables):
MDM [2209.14916](https://ar5iv.labs.arxiv.org/html/2209.14916) ·
MotionDiffuse [2208.15001](https://ar5iv.labs.arxiv.org/html/2208.15001) ·
MLD [2212.04048](https://ar5iv.labs.arxiv.org/html/2212.04048) ·
T2M-GPT [2301.06052](https://ar5iv.labs.arxiv.org/html/2301.06052) ·
MoMask [2312.00063](https://ar5iv.labs.arxiv.org/html/2312.00063) ·
StableMoFusion [2405.05691](https://ar5iv.labs.arxiv.org/html/2405.05691) ·
MotionLCM [2404.19759](https://ar5iv.labs.arxiv.org/html/2404.19759).

**Read on this table:** FID improves roughly 12x from MDM (2022) to MoMask (2024) — the field
moved from diffusion-over-raw-motion toward masked-transformer/VQ-token approaches for the best
numbers, while diffusion variants (StableMoFusion) closed most of the gap. MDM's own FID (0.544)
is a useful sanity floor: **a rebuilt diffusion baseline that cannot beat roughly this number is
not yet in the range where the field's numbers are comparable.**

**R-Precision is saturated at the frontier — flagged by peer review (`reviews/REVIEW_QUEUE.md`
SUP-20260906-02), independently checkable from the table above.** StableMoFusion's R-Prec-top3
(0.841) and MoMask's (0.807) both **exceed the "Real"/ground-truth row's own 0.797.** Generated
motion outscoring real motion on a metric means that metric has no dynamic range left at the
frontier — it can confirm you're in the right regime, but it cannot rank the top of the field.
**Consequence for `REBUILD_SPEC.md`'s D-03 gate: key the harness-validation reproduction on FID,
not R-Precision.** FID still shows an intact, informative range (MDM 0.544 -> MoMask 0.045);
R-Precision does not, at least not among these seven methods.

**Discrepancy flagged, not resolved:** MotionDiffuse self-reports FID 0.681±.001; MLD's own
comparison table cites MotionDiffuse's FID as 0.630±.001. Cross-paper baseline citations
sometimes drift from the source paper's own number (re-runs, eval-code version differences).
Reporting MotionDiffuse's self-reported number as primary; noting the discrepancy rather than
picking silently.

**What was NOT confirmed:** exact numbers for MoMask++/SnapMoGen, MoGenTS, BAMM, MMM, Light-T2M
(search-surfaced names only, no primary fetch); whether MoMask's own table includes a "Real" row;
MotionLCM's KIT-ML numbers (not found in the fetched rendering — "not found" is stated rather
than assumed absent).

### 1.4 What's changed most recently (2025-2026), confidence-flagged

- **SnapMoGen** (arXiv:2507.09122, July 2025, fetched directly) argues HumanML3D's captions are
  the ceiling, not the architecture: HumanML3D averages **12 words/description** vs. SnapMoGen's
  own **48 words/description**. Introduces MoMask++, claimed SOTA on both benchmarks — but no
  numeric HumanML3D table was in the fetched abstract, so MoMask++'s actual numbers are
  UNVERIFIED here.
- A direct critique of the standard protocol exists — arXiv:2411.16575 (CVPR 2025) proposes "a
  more robust evaluation method" per its abstract, but only the abstract was fetched; what
  specifically is claimed wrong with the Guo et al. evaluator is UNVERIFIED pending a full read.
- **ViMoGen** (arXiv:2510.26794, ICLR 2026, [MotrixLab/ViMoGen](https://github.com/MotrixLab/ViMoGen))
  introduces a 228K-sequence dataset and a new benchmark ("MBench") explicitly because, per the
  paper, models still hit "a fundamental bottleneck in generalization capability" when evaluated
  only on HumanML3D-style benchmarks. The most concrete 2025/26 signal that HumanML3D + the Guo
  evaluator remain the shared yardstick but are increasingly seen as insufficient alone.

---

## 2. Text-to-static-pose — datasets actually built for this task

The original course project's real task (text -> one static pose) was attempted on a dataset
built for a different task (text -> motion sequence) — this is F3, separately verified. These
datasets are built for the actual task.

### 2.1 PoseScript (NAVER Labs Europe)

**Task, VERIFIED.** Bidirectional text <-> static 3D pose: text-conditioned pose generation, and
pose-to-text captioning. Operates on individual static poses, not sequences.
[arXiv:2210.11795](https://arxiv.org/abs/2210.11795).

**Size, VERIFIED.** 6,283 human-written descriptions ("PoseScript-H", via Mechanical Turk) plus
~300k auto-generated descriptions (3 each, over 100,000 poses drawn from AMASS).

**Licence, VERIFIED.** **CC BY-NC-SA 4.0** — non-commercial, share-alike.
[github.com/naver/posescript](https://github.com/naver/posescript).

**Availability, VERIFIED.** Both code and data currently downloadable (direct zip link,
pip-installable code).

**Metrics, VERIFIED (paper's Table III).** FID on retrieval-model features, ELBO on
joints/vertices/rotation-matrices, and mRecall in both retrieval directions.

**Its poses are SMPL body-pose parameters** (standard for this literature) — meaning adopting
PoseScript inherits SMPL's licence terms (§4 below), not just PoseScript's own CC BY-NC-SA 4.0.
This was not independently re-verified against PoseScript's data format spec in this pass and
should be confirmed before the spec finalizes — flagged here as a load-bearing assumption.

### 2.2 PoseFix (same lab)

**Task, VERIFIED — not generation from scratch.** Pose *editing*: source pose + text correction ->
target pose, plus the reverse (pose pair -> correctional text).
[arXiv:2309.08480](https://arxiv.org/abs/2309.08480).

**Size, VERIFIED.** 6,157 human-written modifiers, 135,305 auto-generated, 4,284 human
paraphrases. **Licence:** same repo, CC BY-NC-SA 4.0 (no PoseFix-specific clause found distinct
from the repo-wide statement).

### 2.3 PoseEmbroider (NAVER Labs, ECCV 2024)

**VERIFIED.** A genuine shared embedding space across 3D pose, image, and text (transformer core,
special aggregation token) — not just pose+text. Builds on both PoseScript and PoseFix as
dependencies. Some required data (BEDLAM) and weights sit behind separate third-party
registration, so "downloadable" has caveats.
[github.com/naver/poseembroider](https://github.com/naver/poseembroider).

### 2.4 LLM-to-SMPL: ChatPose, and a naming trap

**ChatPose, VERIFIED.** A multimodal LLM (LLaVA/LISA-based) with an SMPL projection layer — the
LLM directly outputs SMPL pose parameters, not just descriptions. CVPR 2024.
[arXiv:2311.18836](https://arxiv.org/abs/2311.18836).

**Naming collision, VERIFIED — worth recording so it isn't rediscovered the hard way.**
ChatPose's repo is now named `github.com/yfeng95/PoseGPT`, self-described as "ChatPose, formerly
known as PoseGPT." This is unrelated to the 2022 ECCV paper *also* called "PoseGPT"
(Lucas et al., VQ-VAE motion-sequence generation/forecasting) — searching "PoseGPT" today
surfaces the wrong one first.

**Benchmark numbers, weakly verified.** ChatPose reports top-5/10/20 retrieval recall of
28.0/39.0/54.4 on classical text-to-pose (RT2P) vs. PoseScript's own baseline at 22.4/32.1/43.6 —
pulled from an arXiv HTML render; the CVF PDF (for cross-check) returned HTTP 403 during
research, so treat these specific numbers as needing a second look before being load-bearing.

**Usability, VERIFIED as concerning.** The repo has 3 commits on main, 13 open issues, and its
`fetch_data.sh` only downloads the SMPL-X body model for visualization — **no trained ChatPose
checkpoint is released.** Research-only, likely not runnable end-to-end without retraining.

**A maintained successor exists.** UniPose (CVPR 2025, arXiv:2411.16781,
[VIPL-VISMOD/UniPose](https://github.com/VIPL-VISMOD/UniPose)) extends the same idea with a pose
tokenizer inside a unified LLM vocabulary, adding pose editing. Has an official repo.
PoseLLaVA and Pose-RFT surfaced via search only — UNVERIFIED, not fetched.

---

## 3. Pose as a control signal — text -> pose -> image/video

### 3.1 Bonnet et al. 2024, "From Text to Pose to Image" (the original paper's own reference [1])

**VERIFIED.** arXiv:2411.12872, NeurIPS 2024 Workshop on Compositional Learning (not main track).
Two-stage pipeline: a custom transformer generates a **2D keypoint pose** (128 points: 18 body +
42 hand + 68 face — DWpose-style, *not* SMPL), then a new pose adapter feeds those keypoints into
SDXL. On a 100-sample COCO-Pose benchmark: beats a CLIP-KNN baseline 78% of the time (CLaPP
score); beats the existing Tencent SDXL-DWpose adapter 70%/76% on aesthetic/HPS-v2. No head-to-
head against ControlNet's own OpenPose conditioning is reported. Code released, MIT licence,
[github.com/clement-bonnet/text-to-pose](https://github.com/clement-bonnet/text-to-pose) —
appears to be a working, if small, released artifact.

### 3.2 The broader pose-conditioning landscape, by maturity

- **ControlNet** (ICCV 2023, [arXiv:2302.05543](https://arxiv.org/abs/2302.05543),
  [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)) — the lowest-effort, most
  battle-tested route. Trainable side-network on a frozen diffusion model, conditions on an
  OpenPose skeleton image among other signal types. SDXL-tuned checkpoints exist and are public
  (e.g. `thibaud/controlnet-openpose-sdxl-1.0` on Hugging Face). **Take a generated pose, rasterize
  it to an OpenPose skeleton image, condition an off-the-shelf pipeline — no training required.**
- **Champ** (ECCV 2024) — renders depth/normal/semantic maps from SMPL sequences plus skeleton
  guidance for video diffusion, richer 3D-aware control than a bare 2D skeleton. Sourced from a
  secondary (Springer) indexing page only, not a direct arXiv/GitHub fetch — architecture/results
  framing here is lower-confidence than the ControlNet/Bonnet entries.
- **Animate Anyone** ([arXiv:2311.17117](https://arxiv.org/abs/2311.17117)) — pose-guided
  image-to-video animation; a ReferenceNet preserves a static reference image's appearance while a
  pose-guider drives motion. No numeric results were visible in the fetched abstract.

**Grounded conclusion:** ControlNet + OpenPose is the lowest-effort way to turn a generated pose
into a demonstrable image today — mature, permissively-checkpointed, zero training. Champ/Animate
Anyone show the field moving toward richer SMPL-aware and temporal conditioning, which would be
more work but a more genuine contribution than re-wiring stock ControlNet.

---

## 4. Body models and tooling

### 4.1 SMPL licence — VERIFIED, quoted

[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) / [licence page](https://smpl.is.tue.mpg.de/modellicense.html):
registration required; use permitted only for "non-commercial scientific research, non-commercial
education, or non-commercial artistic projects." Explicitly prohibited: "incorporation in a
commercial product," "use in a commercial service," training "methods/algorithms/neural
networks/etc. for commercial use." **Redistribution barred**: "shall not be copied, shared,
distributed, re-sold... only one archive copy" permitted. Commercial licensing routes through
Meshcapade.com or smpl@max-planck-innovation.de.

### 4.2 SMPL-X licence — VERIFIED, quoted

[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) — same non-commercial structure, broader
prohibited-use list (adds "military, or surveillance purposes," reverse-engineering ban). Same
redistribution bar. Must cite Pavlakos et al. CVPR 2019. Governing law: Germany / Max Planck.

### 4.3 The `smplx` Python package — licence nuance, VERIFIED

[github.com/vchoutas/smplx](https://github.com/vchoutas/smplx): provides a unified differentiable
PyTorch layer (SMPL/SMPL+H/SMPL-X/MANO) — `model(...)` returns `output.vertices` and
`output.joints` directly, confirmed via the repo's own demo script. **Important:** the *code* is
not separately MIT-licensed — the repo's own LICENSE bundles "Model & Software" under the same
non-commercial-research terms as the model data. Model weights are not bundled; each body-model
family requires its own separate registration (SMPL-X, SMPL+H/MANO, SMPL each have their own
gated download).

### 4.4 A pose/mesh viewer — VERIFIED real and current

[aitviewer](https://github.com/eth-ait/aitviewer) (ETH AIT Lab) — actively maintained, native
SMPL/SMPL-H/SMPL-X/MANO/FLAME sequence support, ModernGL-based ("100fps+ on most laptops"),
pip-installable, with a documented page specifically for working with the SMPL family.

### 4.5 Cost of a differentiable forward-kinematics (FK) layer — VERIFIED, grounded conclusion

`smplx`'s forward pass gives differentiable joints/vertices "for free" **only if** the project
adopts full SMPL/SMPL-X pose parameters — this project's 263-d HumanML3D representation is *not*
SMPL pose parameters, so reusing `smplx` directly would require re-deriving SMPL parameters from
it, which is nontrivial and not free.

**This project already has a working, permissively-licensed alternative.** `primary_source/skeleton.py`
(MIT, fetched from `EricGuo5513/HumanML3D`) implements FK from scratch, with
`forward_kinematics_cont6d(cont6d_params, root_pos, ...)` — taking exactly the 6D continuous
rotation representation already used in HumanML3D's own 263-d vector, built from ordinary torch
tensor operations (`qrot`/`qmul`, differentiable by construction). **This is not new work; it is
already vendored, MIT-licensed, and format-matched to the data this project already has.** Full
SMPL/SMPL-X only becomes necessary if the project wants mesh-level output or PoseScript-ecosystem
compatibility — and that path carries the non-commercial, registration-gated, no-redistribution
licence in §4.1/4.2, inherited by anything built on it.

---

## 5. Text encoders for spatial and lateral language

### 5.1 Is CLIP ViT-B/32's pooled embedding still defensible? VERIFIED weak, from the literature

- ["What's left can't be right"](https://arxiv.org/abs/2311.11477) (arXiv:2311.11477): CLIP-style
  models' left-right/positional incompetence is "entirely predictable" and systematic, not a
  data-scale artifact, though partially fixable with targeted synthetic data.
- ["Left-Right Symmetry Breaking in CLIP-style VLMs"](https://arxiv.org/abs/2601.12809) (ICML
  2026): left-right competence is a fragile, trained-in artifact tied to a specific "horizontal
  attention gradient," not a robust representation. **Scope caveat (peer review,
  `reviews/REVIEW_QUEUE.md` SUP-20260906-01):** this paper is a controlled **1D synthetic
  testbed** — Transformer encoders trained on synthetic spatial-relation data, not CLIP ViT-B/32
  evaluated on real captions. It is mechanistic evidence that this failure mode *can* arise
  under controlled conditions, not direct evidence about how CLIP behaves in the wild. Treated
  here as supporting, not load-bearing — the load-bearing claims about real CLIP models are the
  two below.
- ["Text encoders bottleneck compositionality"](https://arxiv.org/pdf/2305.14897)
  (arXiv:2305.14897): CLIP's **single-vector pooled bottleneck** specifically (not the model in
  general) loses information about object relationships, attribute-object association, counting,
  negation — directly relevant since the original pipeline used exactly this pooled 512-d vector.
- Domain-specific: a text-to-motion survey ([arXiv:2505.09379](https://arxiv.org/html/2505.09379v1))
  notes CLIP-pooled conditioning's documented weakness for fine-grained motion descriptions, and
  that TEMOS/MotionGPT moved to DistilBERT/T5 respectively.

### 5.2 Concrete alternatives, with sourced tradeoffs

1. **CLIP token-level (per-token) embeddings**, not pooled — reuses CLIP's own per-token hidden
   states, costs a cross-attention mechanism in the denoiser instead of a single conditioning
   vector, no new encoder training. Corroborating signal (lower confidence, not independently
   re-verified against the SDXL paper): even image diffusion is moving away from the pooled
   vector toward token-level conditioning.
2. **T5-family encoder** — VERIFIED via direct fetch of the Imagen paper
   ([arXiv:2205.11487](https://arxiv.org/html/2205.11487v1)): "human evaluators prefer T5-XXL
   encoders over CLIP text encoders... on DrawBench" (a deliberately compositional benchmark),
   "perform similarly... on simple benchmarks such as MS-COCO." **Domain-specific counter-evidence,
   directly relevant to this project:** the same motion-generation survey reports that
   researchers who tried swapping CLIP for T5 on a HumanML3D-scale model "did not observe an
   obvious gain," attributing it to HumanML3D's dataset scale being too small to benefit from a
   much larger language encoder. **This is a direct caution against assuming T5 helps here.**
3. **DistilBERT / word-level encoders, already validated in this exact literature.** TEMOS uses a
   frozen DistilBERT and is reported (via a search snippet, not independently confirmed against
   the primary paper — UNVERIFIED) to have the best R-Precision/FID among compared encoders in
   one survey's comparison. Lower-risk than a from-scratch choice since it's already validated on
   HumanML3D-adjacent work.
4. **Generic sentence-transformers** (e.g. `all-mpnet-base-v2`, confirmed real and current via its
   HF model card) — architecturally plausible, but no source was found evaluating these
   specifically on spatial/left-right tasks. UNVERIFIED for this use case.

---

## OPEN_QUESTIONS carried from Part A research

1. ~~Whether PoseScript's poses are literally SMPL pose parameters~~ **RESOLVED (peer review,
   `reviews/REVIEW_QUEUE.md` SUP-20260906-03):** confirmed directly from
   `github.com/naver/posescript` — PoseScript's poses are **SMPL+H G format**, i.e. genuine SMPL
   body-model parameters. The load-bearing assumption in this document and in `REBUILD_SPEC.md`
   was correct: adopting PoseScript does inherit the full SMPL non-commercial licence chain, on
   top of PoseScript's own CC BY-NC-SA 4.0.
2. ChatPose's Table 1 numbers (RT2P/RP2T recall) — sourced from arXiv HTML only, CVF PDF returned
   403. Re-check before quoting elsewhere.
3. Champ's architecture/results — sourced from a secondary indexing page, not a primary fetch.
4. SnapMoGen's MoMask++ numbers, and arXiv:2411.16575's specific evaluator critique — abstracts
   only, bodies/tables not fetched.
5. Whether HumanML3D-derived data (this project's likely dataset) carries the same AMASS/SMPL
   non-commercial restriction as PoseScript — **VERIFIED yes**, separately, by this session
   directly: HumanML3D's own README states "Due to the distribution policy of AMASS dataset, we
   are not allowed to distribute the data directly," and its skeleton follows SMPL's structure
   (`README.md` line 103). This means **neither HumanML3D-derived poses nor PoseScript are clear
   of SMPL/AMASS's non-commercial terms** — the licence question is not "which dataset avoids
   this," it's "both inherit it, so what does that mean for positioning." Carried into
   `POSITIONING.md`.
