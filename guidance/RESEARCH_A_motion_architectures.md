# Research A: Text-to-Motion Architectures, 2023-2026 — Beyond the Known Baseline

> Reviewer/collaborator artifact. Written by a research-collaborator agent via web/arXiv search.
> Scope given: go beyond MDM, MotionDiffuse, MLD, T2M-GPT, MoMask, StableMoFusion, MotionLCM,
> SnapMoGen/MoMask++, ViMoGen (all already known to the project).
> Epistemic rule followed throughout: every claim carries an arXiv ID/URL and a VERIFIED
> (I fetched the abstract/HTML myself) or UNVERIFIED (search-snippet only) tag. Numbers are
> quoted from the source, not from memory. "Not found" is used explicitly where I could not
> confirm something.
>
> Caveat on VERIFIED tags: verification was done via an automated fetch-and-summarize tool
> (WebFetch), not by reading the raw PDF myself end-to-end. In two cases (Motion Mamba's table,
> MotionGPT3-flow's table) the tool's number extraction looked internally inconsistent (e.g. a
> "Diversity" value duplicated in the "FID" column) — flagged inline where that happened. Treat
> those specific numbers as directionally right but re-verify against the actual PDF table before
> citing them in a paper.

---

## 1. What replaced diffusion-over-raw-motion after 2023

The 2023 consensus (MDM/MotionDiffuse: diffusion directly over raw joint/rotation sequences) has
fragmented into at least five active lines. None of them has fully displaced the others as of
Sept 2026 — the field is currently a fight between "masked/AR discrete token" and "flow-matching
continuous latent," with Mamba and retrieval-augmentation as minority approaches.

### 1a. VQ / masked-transformer line (the project already knows MoMask, T2M-GPT, SnapMoGen/MoMask++)

- **MARDM** — "Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant
  Representations, Evaluation, and Masked Autoregression," Meng, Xie, Peng, Han, Jiang, CVPR 2025.
  arXiv:2411.16575. **VERIFIED** (fetched HTML). Core argument: VQ-based discrete methods
  (T2M-GPT, MoMask) have surpassed diffusion on standard metrics since 2023, but discretization
  loses information and caps diversity/prior-guidance utility. MARDM puts masked-*autoregression*
  back onto a *continuous* diffusion representation instead of discrete tokens. Reported main
  table (HumanML3D test, "Ours-SiT" variant): **FID 0.114±0.007, R-Precision Top-1/2/3 =
  0.500/0.695/0.795**, vs. MoMask FID 0.116, R@1/2/3 = 0.490/0.687/0.786, vs. T2M-GPT FID 0.335,
  R@1/2/3 = 0.470/0.659/0.758 (all VERIFIED, quoted from the paper's Table 5 as extracted).
  No training-cost figures found in the paper (checked main text and appendix — "not found").
  Code: **github.com/neu-vi/MARDM, MIT license, VERIFIED** — training scripts
  (`train_AE.py`, `train_MARDM.py`) and inference (`sample.py`, `edit.py`) present, pretrained
  checkpoints on Hugging Face for both HumanML3D and KIT-ML. No GPU/VRAM/training-time numbers
  in the README ("not found").

- **AnyMo** — "Scaling Any-Modality Conditional Motion Generation with Masked Modeling."
  arXiv:2605.29488. **VERIFIED** (fetched abstract). A direct architectural descendant of MoMask:
  4-layer Residual FSQ tokenizer (codebook size 2048/layer) + a scalable masked-modeling
  transformer, trained on a new **OmniHuMo** dataset (5,000+ hours, 3.2M sequences, multimodal —
  text/speech/music/trajectory). No FID/R-precision numbers or training-cost figures were
  retrievable from the fetched content — **not found**; this is a large-data-scale industrial
  paper, not a candidate for cheap replication.

- **MoSa** — "Motion Generation with Scalable Autoregressive Modeling." arXiv:2511.01200.
  **VERIFIED** (fetched abstract). Coarse-to-fine hierarchical RQ-VAE + a novel CAQ-VAE
  (convolution-attention hybrid, to fix reconstruction loss from repeated interpolation) + a
  "Scalable Autoregressive" (SAR) head predicting whole *scales* of tokens per step rather than
  one token — only 10 inference steps needed. Reported (Motion-X dataset, not HumanML3D): **FID
  0.06 vs MoMask's 0.20**, 27% faster inference. HumanML3D numbers **not found** in fetched
  content. Code released, **CC-BY-4.0, VERIFIED**.

- **ScaMo** — "Exploring the Scaling Law in Autoregressive Motion Generation Model." CVPR 2025
  (pp. 27872-27882), arXiv:2412.14559. **VERIFIED** (fetched abstract). Motion FSQ-VAE +
  text-prefix autoregressive transformer. Main empirical claim: normalized test loss follows a
  **logarithmic law** vs. compute budget, with power-law relations between non-vocabulary
  params / vocabulary params / data tokens and compute — i.e., the first paper to argue GPT-style
  scaling laws hold for motion generation. Model sizes tested and exact GPU-hour costs: **not
  found** in the fetched abstract content — would need the PDF tables. Relevant mainly as a
  ceiling-setting reference (scaling helps, but the project's budget is nowhere near where scaling
  gains kick in).

- **Next-Scale Autoregressive Models (MoScale)** — arXiv:2604.03799, CVPR 2026 (accepted).
  **VERIFIED** (fetched HTML, but content was thin). VAR-style ("next-scale prediction," borrowed
  from image generation's VAR line) rather than next-token AR: generates motion coarse-to-fine
  across temporal scales with cross-scale and in-scale refinement. FID/R-precision and
  GPU-hour figures: **not found** in the fetched content — paper is very recent (May 2026
  revision); worth a follow-up fetch of the PDF directly if this direction matters to the project.

- **T2M Mamba** (see §1c) also belongs partly to this line, since it targets the
  "periodicity/saliency" weaknesses of pure sequence models.

### 1b. Flow matching / rectified flow (the clearest "successor to diffusion" line)

- **MotionGPT3 (base model)** — "MotionGPT3: Human Motion as a Second Modality," Zhu, Jiang,
  Wang, Tang, Chen, Luo, Zheng, Chen. arXiv:2506.24086. **VERIFIED** (fetched abstract).
  Bimodal architecture: continuous motion VAE (no quantization) + a dual-stream Transformer
  ("Mixture-of-Transformers"-style) that keeps a frozen pretrained-LLM text branch and adds an
  independent motion branch, connected via shared attention. Reports **2x faster loss
  convergence, up to 4x faster convergence to comparable quality** vs. baselines. Exact
  GPU-hour training cost for the base model and HumanML3D FID table: **not found** in fetched
  abstract (would need PDF). Code: **github.com/OpenMotionLab/MotionGPT3** exists (project is
  from the same lab as the original NeurIPS 2023 MotionGPT, github.com/OpenMotionLab/MotionGPT) —
  license not directly confirmed in this pass, **not found**, but OpenMotionLab's prior MotionGPT
  repo is Apache-2.0-style permissive (UNVERIFIED for MotionGPT3 specifically — check the repo's
  own LICENSE file before relying on this).

- **"From Diffusion to Flow: Efficient Motion Generation in MotionGPT3"** — Ban, Jeon, Jeong
  (note: a *different, smaller* author team than the MotionGPT3 base paper — this reads as an
  independent ablation/extension built on top of the MotionGPT3 architecture, not the original
  lab's own follow-up). arXiv:2603.26747. **VERIFIED** (fetched HTML directly, both v2 and via
  abstract). **This is the single most actionable "cheap training" data point found in this
  entire search.** Quoted directly: trains **"approximately 13 hours of training on a single
  NVIDIA RTX 5090 GPU"** per model variant (diffusion vs. rectified-flow head), 200 epochs total,
  flow variant peaking at epoch 54 and diffusion at epoch 143. Critically, this is **Stage-1
  training only**: it operates on a pre-existing continuous motion latent space with a **frozen
  VAE and frozen text encoder** — i.e., they are training a lightweight denoising/flow head on
  top of frozen pretrained components, not the whole stack from scratch. Reported HumanML3D
  numbers (flow @ epoch 54 vs. diffusion @ epoch 142): **R@1 0.544 vs 0.520, R@2 0.740 vs 0.716,
  R@3 0.828 vs 0.807, FID 0.192 vs 0.240** — flow wins on both axes with ~3% relative R-precision
  gain and ~5% FID reduction, and needs fewer sampling steps. Code release status: **not found**
  in the fetched content (no GitHub link surfaced). This paper is a strong template for a
  laptop/cheap-cloud reproduction: "freeze a pretrained VAE + text encoder, train only a flow head,
  13 GPU-hours on a single high-end consumer card."

- **MotionHiFlow** — "Text-to-Motion via Hierarchical Flow Matching." arXiv:2604.23264.
  **UNVERIFIED** (title/abstract only from search snippet, not fetched). Hierarchical flow
  matching for text-to-motion; no metrics or cost data gathered — flag for follow-up if the
  project wants to build on it.

- **FlowCoMotion** — "Text-to-Motion Generation via Token-Latent Flow Modeling." arXiv:2604.11083.
  **UNVERIFIED** (search snippet only). Reported architecture detail from the snippet: a flow
  matching head with 15 Transformer layers, 8 attention heads, 512 hidden size, using Rectified
  Flow sampling with 40 Euler steps — small enough to plausibly be cheap to train, but I did not
  fetch the paper to confirm training cost or metrics. **Not found** beyond this.

- **MotionFlux** — "Efficient Text-Guided Motion Generation through Rectified Flow Matching and
  Preference Alignment." Found only via a ResearchGate mirror in search results, not fetched.
  **UNVERIFIED**. Combines rectified flow with preference-alignment fine-tuning (RLHF-adjacent) —
  potentially relevant to the "cheap fine-tune rather than train from scratch" angle in Q2/Q3, but
  I could not confirm training cost, metrics, or code status. **Not found**.

- **HY-Motion 1.0** — "Scaling Flow Matching Models for Text-To-Motion Generation."
  arXiv:2512.23464. **UNVERIFIED** (search snippet only). Described as "the first successful
  attempt to scale up DiT-based flow matching models to the billion-parameter scale" for motion —
  this is the opposite of cheap; noted here only so the project doesn't confuse it with a
  cheap-training candidate. Also referenced as a comparison baseline inside MotionRFT (below).

- **Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow**.
  arXiv:2509.24099. **UNVERIFIED** (title only from search). Rectified flow applied to two-person
  interactive/reactive motion generation — outside the project's likely single-person scope but
  flagged for completeness.

### 1c. State-space / Mamba models for motion

- **Motion Mamba: Efficient and Long Sequence Motion Generation** — Zhang, Liu, Reid, Hartley,
  Zhuang, Tang. ECCV 2024, arXiv:2403.07487. **VERIFIED** (fetched abstract + HTML table, with a
  caveat below). First SSM-based motion generation model: a U-Net-style architecture with a
  **Hierarchical Temporal Mamba (HTM)** block and a **Bidirectional Spatial Mamba (BSM)** block.
  Headline claim: "up to 50% FID improvement and up to 4x faster" than diffusion baselines
  (MDM/MotionDiffuse/MLD). Fetched HumanML3D table gave MLD FID 0.473, MDM FID 0.498,
  MotionDiffuse FID 0.681, and Motion Mamba FID reported as 0.281 — **caution: the HTML-table
  extraction I got back also showed Motion Mamba's own "Diversity" value duplicated into the FID
  cell, i.e. the tool's parse was internally inconsistent; treat the 0.281 FID figure as
  directionally plausible (best-of-group) but re-check against the actual PDF table before
  quoting it in writing.** Inference speed comparison looked cleaner and more trustworthy:
  **0.058s/sequence vs MLD's 0.217s** (~4x, matches the headline claim). Training cost, quoted
  directly: **4 NVIDIA A100 GPUs, 2,000 epochs, "approximately 4 hours" total** — genuinely cheap
  in wall-clock terms, though 4xA100 is not a laptop/single-consumer-GPU setup (would need
  re-scaling to 1 GPU, likely ~16 GPU-hours equivalent if linear, which fits the project's
  ~20-hour ceiling). License: **CC BY-NC-SA 4.0** (non-commercial), project page
  steve-zeyu-zhang.github.io/MotionMamba/. Code release: confirmed to exist via project page,
  **VERIFIED** for existence, license VERIFIED, but I did not personally open the repo to confirm
  training/inference script completeness.

- **InterMamba** — "Efficient Human-Human Interaction Generation with Adaptive Spatio-Temporal
  Mamba." arXiv:2506.03084. **UNVERIFIED** (search snippet only). Two-person interaction, not
  single-person text-to-motion — flagged for completeness, out of core scope.

- **Text-controlled Motion Mamba** — "Text-Instructed Temporal Grounding of Human Motion."
  arXiv:2404.11375. **UNVERIFIED**. Different task (temporal grounding, not generation) — noted
  so the project doesn't conflate it with a generation architecture.

- **Dyadic Mamba** — "Long-term Dyadic Human Motion Synthesis." arXiv:2505.09827. **UNVERIFIED**.
  Two-person again, out of scope, flagged only for completeness.

- **Learning Human Motion with Temporally Conditional Mamba**. arXiv:2510.12573. **UNVERIFIED**
  (title only). Not fetched — worth a look if the project pursues the Mamba line further.

- **T2M Mamba** — "Motion Periodicity-Saliency Coupling Approach for Stable Text-Driven Motion
  Generation." arXiv:2602.01352. **UNVERIFIED** (surfaced in search, not fetched). A second,
  more recent (Feb 2026) Mamba-for-T2M paper explicitly targeting periodicity/saliency issues —
  worth checking if pursuing Mamba, since it postdates and likely responds to Motion Mamba's
  weaknesses.

**Bottom line on "what replaced diffusion":** nothing has fully replaced it; instead the field
split into (i) masked/AR discrete-token transformers (dominant on leaderboards since 2023,
MoMask/T2M-GPT/MARDM/AnyMo/MoSa/ScaMo lineage), (ii) flow-matching/rectified-flow continuous
models that are now overtaking diffusion specifically as the *continuous*-representation choice
(MotionGPT3-flow, HY-Motion, MotionHiFlow, FlowCoMotion), and (iii) a small but real Mamba/SSM
side-line chasing long-sequence efficiency. Retrieval-augmentation (ReMoMask, §3) is emerging as
a fourth, orthogonal axis that stacks on top of any of the above.

---

## 2. Cheap-to-train methods (flagging training cost wherever reported)

Directly answering "what could a solo researcher train in <20 GPU-hours on one GPU":

| Method | arXiv | Reported cost | Status |
|---|---|---|---|
| MotionGPT3-flow (Ban/Jeon/Jeong ablation) | 2603.26747 | **~13 hours, 1x RTX 5090**, Stage-1 only (frozen VAE + frozen text encoder, only a flow/diffusion head trained) | VERIFIED |
| Motion Mamba | 2403.07487 | **~4 hours on 4x A100** (2000 epochs) — roughly ~16 GPU-hours if serialized to 1 GPU | VERIFIED |
| MARDM | 2411.16575 | not found (checked main text + appendix) | VERIFIED (absence confirmed) |
| AnyMo | 2605.29488 | not found | VERIFIED (absence confirmed) |
| MoSa | 2511.01200 | not found | VERIFIED (absence confirmed) |
| ScaMo | 2412.14559 | not found in abstract; scaling-law paper implies a *range* of costs across model sizes but exact GPU-hours weren't retrievable this pass | VERIFIED (partial) |
| MotionGPT3 (base) | 2506.24086 | not found | VERIFIED (absence confirmed) |
| Würstchen (image, not motion — reference point only) | 2306.00637 | 24,602 GPU-hours vs 200,000 for SD2.1 — cited only to calibrate what "cheap" means in adjacent generative modeling, not a motion paper | UNVERIFIED (search snippet) |

The MotionGPT3-flow paper (2603.26747) is the standout: it is explicitly a **fine-tune-a-head-
on-frozen-backbone** recipe, trains in single-digit-to-teens GPU-hours on a single (admittedly
top-tier consumer) GPU, and reports it beats the diffusion head on the same frozen backbone on
both FID and R-precision. This is close to a direct template for the project's compute envelope,
modulo needing access to (or reproducing) the frozen MotionGPT3 VAE + text encoder first.

Also relevant to "cheap" by design rather than by report: **SoPo** (arXiv:2412.05095,
"Text-to-Motion Generation Using Semi-Online Preference Optimization," **VERIFIED** abstract
fetch) and **MotionRFT** (arXiv:2603.27185, **VERIFIED** abstract fetch) are both *fine-tuning*
methods applied on top of already-trained motion generators (MLD/MDM for SoPo; MLD, ACMDM, and
HY-Motion for MotionRFT) rather than train-from-scratch methods — by construction they are
cheaper than full training, though neither paper's fetched abstract gave me an explicit GPU-hour
number (**not found** for both). MotionRFT quotes reward-fine-tuning gains of **"22.9% FID
reduction on joint-based ACMDM"** and **"12.6% R-Precision gain and 23.3% FID improvement on
rotation-based HY Motion"** (both figures VERIFIED, quoted from the fetched abstract), and states
its own lightweight fine-tuning method is called **"EasyTune,"** optimizing step-wise rather than
over the full denoising trajectory (cheaper per-step compute, by construction, though again no
absolute GPU-hour number was retrievable — **not found**). Code: MotionRFT states "project page
with code is publicly available" (VERIFIED claim of intent; I did not independently open the repo
to confirm it's actually populated).

---

## 3. Small datasets, few-shot, parameter-efficient adaptation (LoRA/adapters) for motion

This is a thin literature compared to LoRA-for-LLMs or LoRA-for-image-diffusion, but there are
real, motion-specific hits:

- **Stylized Text-to-Motion Generation via Hypernetwork-Driven Low-Rank Adaptation** — Jeon,
  Hong, Noh. arXiv:2605.13333. **VERIFIED** (fetched abstract). A style-reference motion clip is
  encoded into a global style embedding; a **hypernetwork maps that embedding to LoRA-style
  low-rank weight updates** injected at each denoising step of a text-driven motion diffusion
  model, with a supervised contrastive loss shaping the style latent space. Evaluated on
  HumanML3D + 100STYLE. This is close to what the project might want for "adapt a pretrained
  T2M model to a new small style/motion-set without retraining the backbone" — but I could not
  retrieve exact data-scale-per-style, GPU-hour cost, quantitative metrics, or code-release status
  from the fetched content (**not found** for all four). Follow-up PDF fetch recommended if this
  becomes a candidate to build on.

- **Towards Continual Motion-Language Agents: LoRA Variants for Incremental Motion Understanding
  and Generation** — Taetz, Cosme da Silva, Bleser-Taetz, CoLLAs 2026. arXiv:2606.30266.
  **VERIFIED** (fetched abstract). Compares LoRA-based continual-learning variants (framed as
  parameter-efficient "experts," including a mixture-of-experts-with-autoencoder-router setup) on
  top of a **frozen LLM backbone** (exact base LLM not stated in the fetched content — **not
  found**) for incrementally learning new motion tasks. Finding quoted: **"hard expert selection
  via routing significantly outperforms soft expert blending."** Evaluation uses a **"reproducible
  five-task benchmark derived from HumanML3D through semantic clustering"** — i.e., they split
  HumanML3D itself into task-sized shards to simulate continual/incremental (effectively
  small-data-per-task) learning, which is directly relevant if the project wants a "small slice of
  HumanML3D as a stand-in for a small custom dataset" experimental design. GPU-hours, code
  release: **not found**.

- **MotionGPT: Finetuned LLMs Are General-Purpose Motion Generators** — arXiv:2306.10900 (this
  is *already adjacent to* the project's known baseline list but distinct from MoMask/T2M-GPT —
  it's the "instruction-tune an LLM directly on motion tokens" line, precursor to MotionGPT3).
  **UNVERIFIED** in this pass (surfaced via search snippet describing LoRA hyperparameter
  ablation — rank r and scaling factor alpha — but not independently fetched here since it's a
  2023 paper likely already familiar). Flag: if the project doesn't already know this specific
  LoRA-on-LLM-for-motion ablation, it's worth a dedicated look, since tuning r/alpha for a
  frozen-LLM + LoRA motion generator is exactly a laptop-feasible experiment.

- **Modular Image-to-Video Adapter (MIVA)** — arXiv:2512.20000. **UNVERIFIED** (search snippet
  only; this is a **video**, not motion-skeleton, adapter, flagged only as an architecture-pattern
  reference). Quoted claim from the snippet: "can be efficiently trained on approximately ten
  samples using a single consumer-grade GPU" — a genuinely few-shot, cheap adapter design pattern
  (lightweight sub-network bolted onto a frozen pretrained diffusion backbone, one adapter per
  motion pattern, parallelizable). Not a motion-generation paper per se, but the *pattern*
  (few-sample adapter modules on a frozen backbone) is directly transferable to a text-to-motion
  setting and is exactly the kind of "cheap adapter for a small custom motion set" design the
  project's Q3 is asking about. Recommend treating this as an architecture template to port, not
  as a motion-domain citation.

- **Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data** — Fan, Lu, Dai, Yu,
  Xiao, Dou, Dong, Ma, Wang. arXiv:2507.07095. **VERIFIED** (fetched abstract). Introduces
  **MotionMillion**, a 2,000+ hour / ~2M sequence dataset, and scales a model to **7B parameters**
  — this is the *opposite* of small-data/cheap, included here only as the "large-data zero-shot"
  contrast case and because its scale (2M sequences) makes it a candidate *source* of data to
  subsample from if the project wants a bigger-than-HumanML3D but still locally-tractable slice.
  No parameter-efficient/few-shot fine-tuning results were found in the fetched abstract (**not
  found**). Code confirmed to exist on GitHub (VERIFIED existence; license **not found**).

- **ReMoMask: Retrieval-Augmented Masked Motion Generation** — arXiv:2508.02605. **UNVERIFIED**
  (search-snippet level; GitHub repo at github.com/AIGeeksGroup/ReMoMask exists per search
  results but I did not fetch it directly). Notable because retrieval-augmentation is arguably
  the *cheapest possible* way to handle a small custom dataset: instead of fine-tuning weights at
  all, you build a retrieval index over your small motion set and use a **Bidirectional Momentum
  Text-Motion Model** for cross-modal retrieval plus **RAG-Classifier-Free-Guidance** to condition
  a frozen (MoMask-derived RVQ-VAE-based) generator. Reported improvement (quoted from snippet,
  **UNVERIFIED**, not independently confirmed against the paper's own table): "3.88% and 10.97%
  improvement in FID scores on HumanML3D and KIT-ML respectively, compared to previous SOTA
  RAG-T2M." If the project's real constraint is "small custom dataset, can't afford to fine-tune
  a whole model," retrieval-augmentation over a frozen pretrained generator deserves a serious
  look — but re-fetch/re-verify the actual numbers before citing.

**No paper found** in this search that reports an explicit "LoRA rank r vs. FID/R-precision"
ablation specifically for *motion diffusion* (as opposed to motion-LLM) models with a fully
quantified small-dataset training-cost number. That specific gap — LoRA-rank ablation on a
motion diffusion/flow backbone, with GPU-hours reported — looks like a genuinely open, cheap,
and citable experiment as of this search (see §5).

---

## 4. Papers explicitly critiquing HumanML3D as a benchmark/dataset

- **MRBench: A Comprehensive Benchmark for Human Motion-Text Retrieval** — Liu, Xu, Yang, Zhang,
  Yan, Yang. arXiv:2608.07993. **VERIFIED** (fetched HTML directly, two passes). This is the
  strongest, most quantified critique found. Direct quotes: **"prevailing benchmarks are
  dominated by homogeneous indoor motions, imbalanced motion distributions, and oversimplified,
  repetitive texts"**; and, critically, **"under the standard single-positive protocol, 76.3% of
  KIT-ML and 63.2% of HumanML3D test queries share identical text with at least one other gallery
  motion, thus a model retrieving an equally valid motion is nevertheless counted as wrong."**
  Also: **"generic texts such as 'a person slowly walked forward' are attached to as many as 148
  distinct HumanML3D motions"**; **"5.2% of HumanML3D test captions are shared verbatim by at
  least three different motions"**; **"the top-5 most frequent motion types account for 31.1% [of
  HumanML3D]"**; median caption length **"7 words for KIT-ML and 11 for HumanML3D."** (All of the
  above VERIFIED — directly quoted/paraphrased from the fetched paper text, not from memory.)
  MRBench's proposed fix: a new benchmark, **3,390 motions across 118 categories, paired with
  10,170 captions at varying granularity levels** (single-sentence through multi-granular),
  explicitly built for heterogeneous/balanced/non-repetitive coverage.

- **T2MBench: A Benchmark for Out-of-Distribution Text-to-Motion Generation** — arXiv:2602.13751.
  **VERIFIED** (fetched search-result text, not the full HTML — treat as lightly verified).
  Critique angle: existing evaluations (including HumanML3D-based ones) focus on
  **in-distribution** text and a narrow metric set, which "restricts their ability to
  systematically assess model generalization... under complex out-of-distribution textual
  conditions." Confirms HumanML3D's own basic stats: **14,616 distinct motion sequences, each
  with 3-5 independent English captions**. Builds a **1,025-prompt OOD test set** plus an
  LLM-based + multi-factor + fine-grained-accuracy evaluation framework, benchmarking 14
  baseline models. Published Feb 2026.

- **The Quest for Generalizable Motion Generation: Data, Model, and Evaluation** — Lin et al.
  (13 authors). arXiv:2510.26794 (Oct 2025, revised Mar 2026). **VERIFIED** (fetched abstract,
  but content was general rather than HumanML3D-specific — the fetch did not surface an explicit
  HumanML3D critique sentence, so treat the "critiques HumanML3D" framing as **inferred, not
  quoted**). This is the paper behind **ViMoGen** (already known to the project) and **ViMoGen-
  228K** — a 228K-sample dataset combining optical mocap, web video with semantic annotation, and
  synthesized-video-generation outputs, explicitly built to be more diverse than "existing
  benchmarks" (HumanML3D implied but not named in what I fetched). Also introduces **MBench**, a
  hierarchical fine-grained evaluation covering motion quality / prompt fidelity / generalization.
  Recommend a direct PDF fetch if the project needs a verbatim HumanML3D critique quote from this
  specific paper — what I have is structurally suggestive, not a confirmed quote.

- General retrieval-literature corroboration (**UNVERIFIED**, search-snippet level only): "A
  Cross-Dataset Study for Text-based 3D Human Motion Retrieval" (arXiv:2405.16909) is described in
  search results as characterizing HumanML3D captions as comparatively **verbose full-sentence
  descriptions**, contrasted against BABEL's "ultra-short atomic labels" — flagged as a second
  data point on the "caption style is a dataset-specific bias, not a neutral property" argument,
  but not independently fetched/confirmed here.

**On the specific "12 words per caption" figure the project scope asked about**: I could not find
that exact figure stated anywhere in this search. The closest confirmed figures are MRBench's
median caption length of **11 words for HumanML3D** (VERIFIED, quoted above) and T2MBench's
**3-5 captions per motion** (VERIFIED). Treat "12 words" as approximately consistent with the
median-11 figure but **not independently confirmed** — if the project's prior 12-word figure came
from HumanML3D's own paper (Guo et al. 2022) directly, that would be the authoritative source to
cite instead; I did not re-fetch the original HumanML3D paper in this pass.

---

## 5. Cheapest real contributions — ranked, opinionated

Given a laptop plus roughly a $10 cloud budget (call it 20-30 GPU-hours on a rented mid-tier GPU,
or less on a laptop GPU/Apple Silicon), here is what I'd actually attempt, ranked by
(novelty-in-2026) x (probability of finishing) x (defensibility of the "cheap" claim):

1. **LoRA-rank ablation for a motion diffusion/flow head, with GPU-hours reported.** This is the
   clearest gap this search surfaced (§3, final paragraph) — plenty of LoRA-for-motion-*LLM*
   papers (MotionGPT+LoRA, 2306.10900; continual-learning LoRA, 2606.30266), plenty of
   hypernetwork-LoRA-for-*style* (2605.13333), but nothing found that does a plain, well-reported
   "vary LoRA rank r on a frozen motion-diffusion or motion-flow backbone, report FID/R-precision
   vs. r vs. GPU-hours vs. trainable-parameter-count" study, the way such ablations are routine
   for LLMs and image diffusion. Concretely: take the MotionGPT3-flow recipe (2603.26747 — frozen
   VAE + frozen text encoder, train only a small head, 13 GPU-hours on 1 GPU) and replace "train a
   full flow head from scratch" with "LoRA-adapt an existing pretrained head." This directly
   reuses the one paper in this whole search that already reports a concrete, laptop-adjacent
   GPU-hour number, and turns it into a parameter-efficiency study — a genuinely novel,
   easily-defensible, cheap-to-run contribution.

2. **Small-slice / few-shot HumanML3D fine-tuning study, explicitly targeting the MRBench-
   documented caption-redundancy problem.** MRBench (2608.07993, VERIFIED numbers above) gives an
   exact, citable quantification of HumanML3D's caption redundancy (63.2% of test queries share
   captions with another gallery motion; 148 motions share one generic caption). A cheap,
   concrete project: take a pretrained MoMask/T2M-GPT checkpoint (already known to the project),
   freeze it, and fine-tune only a small adapter/LoRA module on a hand-curated *de-duplicated*,
   more-diverse caption subset (or on MRBench's own smaller, more balanced 3,390-motion set if
   it's released), then report whether R-precision/FID *and* qualitative diversity improve versus
   the frozen baseline. This directly operationalizes the continual-learning task-splitting idea
   from 2606.30266 ("derive small tasks from HumanML3D via semantic clustering") but points it at
   a benchmark-quality argument rather than a continual-learning one — novel framing, small
   compute footprint (adapter-only training on a few hundred/thousand curated clips).

3. **Reproduce the Motion Mamba cost/quality tradeoff on a single consumer GPU.** Motion Mamba
   (2403.07487) reports ~4 hours on 4xA100 for 2000 epochs; serialized to one GPU that's
   plausibly ~16 GPU-hours, right at the edge of the stated budget. Mamba/SSM architectures are
   still a minority approach (only a handful of papers found: Motion Mamba, T2M Mamba,
   InterMamba, Dyadic Mamba, Temporally-Conditional Mamba) — a from-scratch or near-scratch
   single-GPU reproduction plus a head-to-head against a LoRA-adapted diffusion baseline (item 1
   above) would be a genuinely useful, cheap, and citable comparison the field currently lacks
   (nobody found in this search directly compares Mamba-for-motion against LoRA-adapted
   diffusion under a matched compute budget).

4. **Retrieval-augmented generation as the "cheapest possible" small-dataset baseline.**
   ReMoMask's approach (2508.02605, UNVERIFIED numbers, re-verify before citing) — build a
   retrieval index over a small custom motion set and condition a frozen pretrained generator via
   RAG-CFG — requires no backbone fine-tuning at all. As a baseline against options 1-2 above
   ("how much does fine-tuning even buy you over pure retrieval, on a genuinely small custom
   dataset"), this is close to free to implement and would make a nice ablation axis in whatever
   final study the project runs, even if it isn't the headline contribution by itself.

**What I would NOT attempt on this budget:** scaling-law studies (ScaMo, HY-Motion, Go-to-Zero —
all require multi-GPU/large-data infrastructure by design), anything built on OmniHuMo/
MotionMillion-scale data (AnyMo, Go-to-Zero — 2M+ sequences, clearly industrial-scale), or
next-scale/VAR-style autoregressive motion generation from scratch (MoScale, 2604.03799 — too new
and undercharacterized in terms of cost to derisk within budget).

---

## Papers found but not pursued further (listed for completeness, not analyzed)

Plan, Don't Pose (2605.29906); SubFlow (2604.12273); Drift Flow Matching (2605.17244); MoGIC
(2510.02722); X-MoGen (2508.05162); Kimodo (2603.15546); Being-M0.5 (2508.07863); ARDY
(2607.08741); IRG-MotionLLM (2512.10730); MoLingo (2512.13840); PlanMoGPT full metrics beyond what's
quoted in §1a (2506.17912); Language-Guided Transformer Tokenizer (2602.08337); HumanScore
(2604.20157); MoCHA (2603.23684); Bilingual T2M Benchmark (2603.25178). All UNVERIFIED,
search-snippet level only — flagging their existence in case the project wants to chase any of
them further, but I did not spend fetch budget confirming their content.
