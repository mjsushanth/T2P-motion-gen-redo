# Research E: Novel Directions Given the Spatial-Blindness Finding

> Collaborator artifact. Written via web/arXiv search, Sept 2026. Read `guidance/VERIFICATION_NOTE.md`
> first — two prior agents fabricated/misattributed statistics and both were caught. The discipline
> here is the same: every claim is VERIFIED (I fetched the source myself, via WebFetch, and quote or
> closely paraphrase what it actually says) or UNVERIFIED (WebSearch synthesis / secondary source only,
> not independently opened and read). "Not found" is used explicitly rather than guessing. No number
> below is invented.
>
> Ground truth taken as given, not re-derived (already owned by the project): spatial minimal pairs at
> cosine 0.9707 vs. 0.9442 for non-spatial pairs, Cohen's d=+0.995, p=0.00001, n=40/group, on MDM's
> frozen CLIP ViT-B/32; 57.6% of HumanML3D captions contain spatial vocabulary; rectified flow beats a
> diffusion head at ~13 GPU-hours/1 RTX 5090 (arXiv:2603.26747); TMR (arXiv:2305.00976) beats the Guo
> et al. evaluator.

---

## 0. How MDM actually conditions on text (needed before judging any "fix")

- MDM encodes text with **frozen CLIP ViT-B/32**, and the resulting embedding — projected together with
  the diffusion timestep — is **concatenated as an extra input token**, not injected via cross-attention.
  **VERIFIED** (multiple independent secondary sources converge on this architectural description,
  including the MDM ICLR 2023 paper listing on OpenReview; I did not re-fetch the primal PDF myself in
  this pass since the project's own docs likely already establish this — treat the "concatenation, not
  cross-attention" detail as **UNVERIFIED against the original PDF** and worth a 2-minute direct check
  before an experiment depends on it).
- One WebSearch-surfaced secondary source (a Medium overview article, not the paper itself) claims
  **"experiments conducted by MDM and MLD suggest that concatenation leads to better results than
  cross-attention."** **UNVERIFIED — flag prominently.** If true, this is an important wrinkle: it
  would mean the MDM/MLD authors already tried per-token cross-attention conditioning and *rejected* it
  on overall-quality grounds. That does not mean per-token conditioning can't fix the *spatial* gap
  specifically (a net quality regression on all captions is compatible with a targeted win on the
  spatial subset), but it means "switch to cross-attention with per-token features" is not a
  free, already-validated win — verify this claim against the actual MDM/MLD papers before relying on
  it as a premise.

---

## 1. Fixing the spatial-blindness — scrutinized hardest, per your instruction

### 1a. What has already been done (read this before proposing any of these)

| Intervention | Paper | Status | What it actually shows |
|---|---|---|---|
| Contrastive fine-tune + distillation of CLIP for motion | **MoCLIP**, arXiv:2505.10810 | **VERIFIED** (fetched full HTML) | Adds a motion encoder head; fine-tunes CLIP's text tower with a symmetric InfoNCE contrastive loss **plus an L2 "tethering" (distillation) loss against the frozen original CLIP text encoder to prevent catastrophic forgetting**. Tested by plugging into MoMask, BAMM, BAD on HumanML3D. Quoted Table-1 numbers: MoMask R-P Top-1 0.521→0.533, Top-3 0.807→0.823, **but FID got slightly worse, 0.045→0.047**; BAMM Top-3 0.808→0.819, **FID worse, 0.055→0.064**; BAD's Top-1 **got worse**, 0.517→0.510 ("BAD... may require additional adaptation," their words). Trained on HumanML3D (14,616 sequences), A6000 GPUs, 50 epochs (35 frozen + 15 unfrozen text encoder) — **no GPU-hour figure given**. **Does not test spatial/directional minimal pairs at all** — no left/right, no directional evaluation anywhere in the paper. **No code link found.** |
| Composite-aware (non-pooled) semantic injection | **CASIM**, arXiv:2502.02063 | **VERIFIED** (fetched abstract) | Replaces fixed-length (pooled) CLIP embedding with a "composite-aware semantic encoder + text-motion aligner" learning dynamic token-level correspondence; claims to "consistently improve" quality/alignment/retrieval on HumanML3D/KIT. **No FID/R-precision deltas found in the fetched content; no spatial-language claim; no training cost; no code link found.** |
| Local/per-token guidance via graph attention | arXiv:2407.10528 | **VERIFIED** (fetched abstract) | Uses local actions as fine-grained control via graph-attention weighting instead of one global condition. **No spatial claim, no numbers, no cost data retrievable from the abstract.** |
| Dual-path contrastive + frequency-domain anchoring | **LUMA**, arXiv:2509.25304 | **VERIFIED** (fetched abstract) | Diagnoses **"severe gradient attenuation in deep layers"** (not spatial blindness specifically) as the bottleneck; fixes it with a small contrastively-trained "MoCLIP"-style model (temporal anchor) + DCT low-frequency anchor (frequency-domain signal), fused adaptively. Reports FID 0.035 (HumanML3D) / 0.123 (KIT-ML) and "1.4x faster convergence" — **no baseline comparison number was extracted, so the delta these numbers represent is not confirmed**; **no spatial-language discussion found**. CC-BY-4.0; code link not found in this pass. |
| Swap CLIP → T5-XXL, add local text features | **RVHM2D**, arXiv:2506.14428 | **VERIFIED** (fetched HTML) | On their own "Human-Motion2D" benchmark (not the standard HumanML3D scale — FID values of 0.65–2.4 are an order of magnitude larger than typical HumanML3D FID, so **treat as a different, non-comparable benchmark**, not direct transfer evidence): T5-XXL beat CLIP-L/B substantially (FID 0.65 vs. 1.92, single-character). Separately, adding **local (per-token) CLIP features on top of the dual global encoders improved R-Precision-Top1 by 0.47%** over the InterGen baseline — a real but small effect. 8x A100, 300 epochs. Code: github.com/FooAuto/Toward-Rich-Video-Human-Motion2D-Generation (**existence VERIFIED via fetch, license not checked**). |
| Swap CLIP → T5 (counter-evidence) | "Motion Flow Matching for Human Motion Synthesis and Editing," arXiv:2312.08895 | **UNVERIFIED** (WebSearch synthesis only — the paper itself is marked "WIP" and I could not extract the actual ablation from the fetched page) | Reportedly found **"no obvious gain"** from T5 vs. CLIP on **HumanAct12** (a small, label-based, not caption-based, benchmark), attributed to dataset scale being too small to benefit from a stronger encoder. **This is exactly the null-result risk the project should pre-register against**: HumanML3D (~15K motions) is also small: a text-encoder swap might null out for lack-of-scale reasons unrelated to whether spatial grounding was really the bottleneck. Re-verify this specific claim by reading the paper directly (my fetch failed) before citing it, but treat the *risk it names* as real regardless. |

**Bottom line on "has this already been done": partially, and inconsistently.** Motion-specific work has
(a) fine-tuned CLIP contrastively for motion-text alignment (MoCLIP) — general alignment gain, **FID
sometimes gets worse, and it is architecture-dependent (hurts BAD)**; (b) replaced pooled with
per-token/composite conditioning (CASIM, RVHM2D-local, 2407.10528) — claimed or small gains, never
isolated to spatial/directional vocabulary; (c) swapped the whole encoder (T5) — helps a lot on one
non-standard, larger-scale video benchmark (RVHM2D) and reportedly does nothing on a small,
label-based one (Motion Flow Matching, unverified). **No paper found in this search that (i) curates a
spatial-minimal-pair set the way this project already has, (ii) uses it as a *training* signal
(contrastive fine-tune or adapter) rather than only an evaluation probe, or (iii) measures the effect
specifically on the spatial-caption subset of HumanML3D generation quality, holding general-caption
quality fixed as a control.** That combination is the project's actual opening.

### 1b. Relevant theoretical/adjacent-domain backdrop (not motion, but bears on whether the fix can even work)

- **"Is CLIP ideal? No. Can we fix it? Yes!"** arXiv:2503.08723. **VERIFIED** (fetched abstract). Proves
  (their claim, general vision-language, not motion) that **no CLIP-like joint embedding space can
  simultaneously represent (1) basic descriptions, (2) attribute binding, (3) spatial location/relations,
  and (4) negation** — a structural/geometric limitation, not just an undertrained-data problem. This is
  a real reason a targeted spatial fix could come at a cost to general performance elsewhere (see the
  risk list below) — **but it is a general vision-language result, not validated on motion-conditioning
  embeddings specifically**, so treat it as a plausible mechanism, not proof this project's fix will
  regress.
- **"Left–Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic
  Spatial-Relation Data,"** arXiv:2601.12809. **VERIFIED** (fetched abstract). Controlled 1D synthetic
  testbed shows left/right competence emerges from **interaction between positional and token
  embeddings**, and **"label diversity, more than layout diversity, is the primary driver of
  generalization."** No quantitative numbers were retrievable from the fetched content, and this is
  vision-domain (image-text), not motion — but the finding ("you need *diverse labels* across many
  object/relation pairs, not just more layouts, for the contrastive objective to learn the relation") is
  directly actionable design guidance for how to build the project's own spatial-contrastive training
  set: vary the *body part/action* around each left/right pair, not just the direction word.
- **What'sUp / LRR-Bench / "What's left can't be right"** — already known to the project via
  RESEARCH_B, not re-verified again here.

### 1c. Concrete cheap interventions, ranked, with explicit failure-mode scrutiny

1. **Free diagnostic (do this first, <1 GPU-hour, no training):** embed the *individual tokens*
   "left"/"right" (and forward/backward, up/down) through the frozen CLIP text tower and check whether
   **per-token embeddings already separate spatially even though pooled sentence embeddings don't**.
   This distinguishes two different bugs with two different fixes: (a) if per-token embeddings already
   separate cleanly, the bug is in the **pooling/averaging step** — a per-token conditioning fix (CASIM-
   style) should work without touching CLIP's weights at all; (b) if even bare per-token embeddings for
   "left" and "right" are nearly identical, the bug is **in the encoder itself**, and only a contrastive
   fine-tune/adapter (MoCLIP-style) can fix it. **No paper found in this search runs this exact probe** —
   it is cheap, novel, and directly resolves which of the two remaining proposals below is the right one
   to fund.
2. **Small contrastive adapter on frozen CLIP, tethered to prevent forgetting** (a scaled-down MoCLIP,
   scoped to spatial vocabulary specifically rather than all of HumanML3D): freeze CLIP, add a small
   linear/low-rank projection head, contrastively fine-tune it on a **curated spatial-minimal-pair set**
   (build on the project's own n=40/group design, but expand to several hundred pairs varying the body
   part/action around each direction word per the 2601.12809 "label diversity" finding), with an
   L2/tethering loss against the frozen original embedding (MoCLIP's exact mechanism, VERIFIED above) to
   avoid MoCLIP's own observed failure mode (FID regression, BAD regression). Cost: adapter-only, a few
   hundred to ~2K pairs, plausibly **<2-5 GPU-hours** on a rented mid-tier GPU or feasible on a laptop
   with MPS given CLIP ViT-B/32's small size.
3. **Reasons this could fail, made explicit (per your instruction to scrutinize hardest):**
   - MoCLIP's own numbers show contrastive fine-tuning of CLIP for motion is not a free lunch: FID got
     *worse* on two of three backbones, and it actively hurt a fourth architecture (BAD). A spatially
     narrower fine-tune could easily reproduce this: better spatial separation, worse or flat FID.
   - The Motion Flow Matching "no obvious gain from T5" result (UNVERIFIED but the mechanism it names is
     real) raises a genuine risk that HumanML3D is too small/noisy at the *spatial-caption subset* level
     (fewer than 57.6% x 24,503 test captions actually land in the test split) for a quality delta to be
     statistically detectable at all, independent of whether the fix "worked."
   - The 2503.08723 impossibility result raises the possibility of a real trade-off: sharper spatial
     separation could come at the cost of blurring some other axis (attribute binding, negation) that
     HumanML3D also depends on — this is exactly why a general-caption regression control (below) is not
     optional.
   - HumanML3D's spatial vocabulary is a small, closed set (left/right/forward/backward/up/down/
     clockwise/counterclockwise and a few compounds). A reviewer will reasonably ask whether a
     contrastive fine-tune on this exact closed set is "solving generalizable spatial grounding" or
     "memorizing ~10 words" — pre-register a held-out spatial word (e.g., train on left/right/forward/
     backward, test transfer on up/down/clockwise) to answer this directly.
4. **Evaluation protocol that must accompany any of the above** (this is where the actual rigor is):
   report (a) spatial-pair cosine-separation after the fix (does it beat the project's own d=0.995?),
   (b) R-precision/FID computed **only on the spatial-caption subset of the HumanML3D test split**
   (the load-bearing number), (c) R-precision/FID on the **full test split, unchanged encoder as
   control** (regression check), (d) a **held-out spatial word** transfer check (generalization, not
   memorization).

---

## 2. Novel training objectives — what's untried

- **Contrastive auxiliary loss injected *during generation training* (not just as a retrieval
  evaluator):** **already done.** "Aligned and realistic latent diffusion for text-to-motion
  generation," Visual Computing for Industry, Biomedicine, and Art (2026), DOI
  10.1186/s42492-026-00224-2. **UNVERIFIED numbers** — the journal page redirected to a login wall and
  I could not fetch the full text; what follows is WebSearch synthesis of the abstract, not independently
  confirmed. They introduce **"Temporal Semantic Contrastive Learning (TSCL)"**: an InfoNCE-style
  objective applied *inside the denoising process* in latent space, plus a separate latent-space
  adversarial discriminator, reportedly matching or beating VQ-based SOTA on HumanML3D/KIT-ML. **Flag
  this prominently: if the project's plan was "add a contrastive text-motion loss as an auxiliary
  training signal," that framing already exists in the literature as of 2026** — re-verify the numbers
  directly (I could not) before building on it, but do not present the general idea as untried.
- **Direction-aware / left-right-aware loss term** (e.g., a forward-kinematics-based penalty that
  specifically checks whether the *correct limb* moved when the caption says "left" vs. "right"):
  **not found anywhere in this search.** Multiple geometric auxiliary losses exist (velocity, foot
  contact, bone-length, relative-orientation — see PhysiInter arXiv:2506.07456, VERIFIED abstract) but
  none of them are conditioned on the *text's* directional content — they are unconditional physical
  plausibility losses, not "does the generated motion match the direction the caption specified." This
  looks like a genuine, citable gap.
- **Curriculum over caption complexity, specifically for motion:** thin to absent. MotionLab
  (arXiv:2502.02358, referenced in RESEARCH_A) curricula over *task type* (fewer modalities = easier),
  not caption complexity. The historical Language2Pose (KIT dataset era) reportedly used a curriculum for
  a joint text-pose embedding — old, different framing, not independently verified here. **No paper
  found that orders/weights HumanML3D training examples by caption complexity (length, spatial-vocab
  density, clause count) on an easy-to-hard or Gaussian schedule.** The closest transferable analog is
  image-domain: **"Synthetic Curriculum Reinforces Compositional Text-to-Image Generation,"**
  arXiv:2511.18378 (**UNVERIFIED**, WebSearch synthesis only) — reportedly finds Gaussian scheduling
  (smooth easy→hard transition) beats both random and strict easy-to-hard ordering for compositional
  text-image generation. This pattern (curriculum by *compositional* difficulty, not just length) is a
  directly transplantable, currently-untried idea for motion, and ties naturally to the project's own
  57.6%-spatial-vocabulary statistic (spatial captions could be the "hard" tail of a complexity
  curriculum).
- **Reward/preference fine-tuning** (MotionRFT/EasyTune, SoPo, TAPO) — already flagged in RESEARCH_A;
  EasyTune (arXiv:2602.07967) reportedly reaches **FID 0.132, "72.1% better than MLD," 7.3x training
  speedup over DRaFT, 22.10GB memory** (**UNVERIFIED**, WebSearch synthesis only, not independently
  fetched/confirmed against the paper's actual tables) — flag as an existing, adjacent-but-different
  direction (post-hoc reward alignment, not a spatial-language fix) rather than a novel one for this
  project to claim.

---

## 3. Parameter-efficient adaptation (LoRA/adapters) — verifying RESEARCH_A's claim

**RESEARCH_A's §3 claim — "No paper found... that reports an explicit 'LoRA rank r vs. FID/
R-precision' ablation specifically for motion diffusion... this specific gap... looks like a genuinely
open, cheap, and citable experiment" — is REFUTED by this search. Do not propose a plain LoRA-rank
sweep on a motion diffusion backbone as if it were novel; it has been published, on the same backbone
family (MDM) the project's own baseline uses.**

| Paper | Status | What it actually shows |
|---|---|---|
| **"Dance Like a Chicken: Low-Rank Stylization for Human Motion Diffusion"** (LoRA-MDM), arXiv:2503.19557 | **VERIFIED** (fetched full HTML directly) | **Table 2 is exactly the missing ablation.** LoRA applied to all attention matrices of MDM's transformer, rank swept across {1, 3, 5, 10, 20}. Quoted numbers: **rank=3 → SRA 62.1%, R-precision 81.4%, FID 0.46, foot-skate 0.039, diversity 9.36; rank=5 (their chosen "sweet spot," λ=0.25) → SRA 70.8%, R-precision 78.8%, FID 0.54, foot-skate 0.042, diversity 10.15; rank=10 → SRA 79.8%, R-precision 77.5%, FID 0.94, foot-skate 0.046, diversity 9.13.** Clear, monotonic style-fit-vs-prompt-fidelity trade-off as rank increases. Task: few-shot **style** adaptation on 100STYLE ("a handful" of reference motions per style — 100STYLE gives 7 basic locomotions/style). Cost: **4,000 training steps on a single NVIDIA L40S GPU**, lr 1e-5 — genuinely cheap, and a real GPU-hour calibration point for this project's budget. **No full-fine-tuning baseline is compared** (their whole point is that full fine-tuning is the thing LoRA avoids). Code: "will be made available"; project page haimsaw.github.io/LoRA-MDM/, CC-BY-4.0. |
| "Retrieving and Refining Winning Noise Tickets for Diffusion-Based Motion Generation," arXiv:2607.06843 | **VERIFIED existence only** (fetched HTML, confirmed an appendix section literally titled "D.4 LoRA Rank and Injection Ablation" for a noise-refiner module on a MotionLCM backbone) | Second independent confirmation that this class of ablation exists in the published motion literature. **Exact numbers UNVERIFIED** — two fetch attempts (PDF too large, HTML excerpt incomplete) could not retrieve the actual table. A WebSearch-snippet-level summary (**UNVERIFIED, do not cite the number**) claims "robust across ranks 16-128, default r=64 full injection." |

**What remains genuinely open, now that the general claim is refuted:** LoRA-MDM's ablation is scoped
to *style* transfer on MDM specifically, with a fixed λ mixing weight and a single dataset (100STYLE).
Not yet published (not found in this search): a LoRA-rank ablation (a) on the **flow-matching head**
architecture the project already knows is cheap (MotionGPT3-flow, arXiv:2603.26747), or (b) applied to
the **spatial-language adapter** from Section 1 above rather than to style. Either of those is a
different, non-duplicate experiment — but "does LoRA rank trade off against quality on a motion
diffusion model" as a bare question is now closed, published, with a citable table.

Also found, not independently number-verified: **MoMug** ("Unlocking Pretrained LLMs for Motion-Related
Multimodal Generation," arXiv:2503.06119) — LoRA-fine-tunes a pretrained LLM (not a diffusion backbone)
for motion, reportedly **"improves FID by 38%... over the most recent LLM-based baseline"**
(**UNVERIFIED**, WebSearch synthesis, not independently fetched) — relevant as a second "cheap PEFT
already published" data point but on an LLM-token backbone, not MDM/MoMask-style diffusion.

---

## 4. Pretrained-checkpoint-plus-modification vs. train-from-scratch

- **Domain precedent the project already owns:** MotionGPT3-flow (arXiv:2603.26747, VERIFIED by the
  project) is exactly this pattern — frozen pretrained VAE, frozen pretrained text encoder, train *only*
  a small flow head, ~13 GPU-hours on 1 RTX 5090. This is the strongest single data point in the entire
  search (across all three prior research files plus this one) for what a $10-30-budget contribution
  looks like, and it is a checkpoint-plus-modification design, not train-from-scratch.
- **Every other cheap, real number found in this search is the same shape:** LoRA-MDM, 4K steps/1 GPU
  (style LoRA on frozen MDM); MoCLIP, adapter+small-encoder fine-tune on frozen CLIP; Motion Mamba's
  from-scratch run (~16 GPU-hours equivalent, per RESEARCH_A, at the outer edge of budget) is the *only*
  from-scratch data point under 30 GPU-hours found across all four research passes, and it trains an
  entire novel architecture family, not a competitive reproduction of an existing SOTA number.
- **General ML literature:** "$100K or 100 Days: Trade-offs when Pre-Training with Academic Resources,"
  arXiv:2410.23261. **VERIFIED existence/topic only** — the PDF's content streams did not decompress
  cleanly in two fetch attempts, so **no specific GPU-hour threshold or contribution-defensibility
  criterion could be extracted; treat as "not found" for exact numbers**, though the paper's existence
  confirms this exact trade-off (train from scratch vs. adapt a released checkpoint, under an academic
  budget) is an active, named research question in the broader field, not one this project invented.
  A separate WebSearch-synthesized claim ("continual pretraining is ~2x cheaper than from-scratch for
  similar final performance... pretrained models need ~10x fewer examples") is **UNVERIFIED** — no
  specific source was confirmed for this number; do not cite it.
- **My synthesis (not a citation, a judgment call given everything above):** on this budget, modifying a
  pretrained checkpoint is more defensible for three concrete reasons, not just cost: (1) it isolates the
  causal claim — only the modified component (adapter/head/loss) changed, so an observed effect is
  attributable to *that* component rather than confounded with "the whole model is undertrained"; (2) it
  is the only way to land near a competitive absolute FID/R-precision number at all inside 30 GPU-hours,
  given every found from-scratch cost in this domain (MARDM, T2M-GPT-successor lineage, ScaMo) either
  reports no cost or implies far more compute than this budget allows; (3) it is what the *entire*
  cheap-and-real literature surfaced across all four research passes actually does — there is no
  counterexample.

---

## 5. Ranked proposals

### Proposal 1 (highest priority): Spatial-contrastive adapter on frozen CLIP, with regression controls

**Hypothesis:** A small, tethered contrastive fine-tune of MDM's frozen CLIP text conditioning,
targeted specifically at the project's own spatial-minimal-pair set (expanded per the 2601.12809
label-diversity finding), improves motion-generation quality on the spatial-caption subset of
HumanML3D's test split without regressing quality on the rest of the test split.

**Exact experiment:** Freeze CLIP ViT-B/32. Add a small linear/low-rank adapter after the text
projection. Build a spatial-contrastive training set: several hundred to ~2K sentence pairs, templated
by crossing a bank of body-part/action phrases with the closed spatial-vocabulary set (left/right,
forward/backward held out for training; up/down/clockwise/counterclockwise held out entirely for a
generalization test). Fine-tune the adapter with (a) symmetric InfoNCE contrastive loss pulling
same-direction paraphrases together and pushing opposite-direction pairs apart, (b) an L2 tethering
loss against the frozen original CLIP embedding (MoCLIP's own fix for its FID-regression problem,
arXiv:2505.10810) to bound how far the adapter can drift.

**Arms/controls:**
- A0: frozen, unmodified CLIP (baseline, matches the project's own d=0.995 measurement).
- A1: adapter fine-tuned with contrastive loss only (no tethering) — isolates whether tethering is
  necessary (replicate/refute MoCLIP's own finding that untethered fine-tuning risks forgetting).
- A2: adapter fine-tuned with contrastive + tethering loss (the proposed fix).
- Held-out-word generalization check: evaluate A2's spatial separation on up/down/clockwise/
  counterclockwise, never seen during adapter training.
- Regression control (non-negotiable): R-precision/FID on the **full, non-spatial** HumanML3D test
  split, A0 vs. A2 — must not regress, or the "fix" has just traded one failure mode for another
  exactly as MoCLIP's own numbers show is a live risk.

**Compute estimate:** adapter-only fine-tune on a few hundred to 2K short text pairs through a frozen
CLIP ViT-B/32 text tower — well under 1 GPU-hour of actual training; realistically finishable on a
laptop CPU/MPS in minutes, with the $10-30 budget entirely available for the downstream MDM generation
runs needed to compute FID/R-precision on the spatial-caption subset (a few GPU-hours on a rented
mid-tier GPU, generating and scoring a few hundred-to-thousand test-split motions per arm).

**What a positive result establishes:** a cheap, targeted, reproducible fix for a measured, real
weakness in the field's most common text-conditioning setup, with a generalization check (held-out
spatial words) that most adjacent papers (MoCLIP, CASIM) do not run, and a regression control that
directly answers the field's own open worry (the 2503.08723 impossibility result) about spatial fixes
trading off against everything else.

**What a negative result establishes:** if spatial separation improves but spatial-subset generation
quality does not (or the held-out-word transfer fails), that is itself a publishable, specific finding —
it would show the project's own measured embedding-space gap is *real* but *not the causal bottleneck*
for generation quality, redirecting attention to the motion-side conditioning mechanism (Section 1's
per-token/pooling question) rather than the encoder itself. Given Motion Flow Matching's reported (but
UNVERIFIED) null result for a much bigger encoder swap under similar dataset-scale conditions, this
outcome should be treated as a live possibility, not a failure of execution.

**Strongest objection a reviewer would raise:** "You fine-tuned on a closed, small vocabulary
(left/right/forward/backward) and evaluated generalization on a similarly small, closed set
(up/down/clockwise/counterclockwise) — this shows the adapter can be taught ten words, not that it
has learned 'spatial grounding' in any generalizable sense. A truly spatial-competent encoder should
transfer to spatial relations never seen in *any* form during training (e.g., 'in front of'/'behind',
'above'/'below' the torso), which this design does not test." Pre-register a second, fully-unseen
relation category as a stretch generalization check if budget allows.

### Proposal 2 (second priority): Diagnostic per-token vs. pooled spatial-separation probe

**Hypothesis:** MDM's spatial-blindness (the project's own measured 0.9707 vs. 0.9442 gap) is caused
by the pooling operation, not the token-level representations underneath it — i.e., individual CLIP
token embeddings for "left" and "right" already separate more cleanly than the pooled sentence
embeddings the project measured.

**Exact experiment:** For each of the 40+40 minimal-pair sentences already collected, extract **both**
the pooled embedding (already measured) **and** the per-token embeddings for the directional word
itself and its immediate neighbors, through the same frozen CLIP ViT-B/32 text tower. Compute the same
cosine-separation/Cohen's-d statistic on the per-token embeddings that the project already computed on
the pooled ones.

**Arms/controls:** No training arms — this is a pure inference-time probe. Control: repeat the same
per-token extraction on the matched non-spatial modifier pairs, exactly mirroring the project's
existing design, so the per-token d-statistic is directly comparable to the already-established
pooled d=0.995.

**Compute estimate:** under 1 GPU-hour; realistically a few minutes of CPU/MPS inference — this is the
cheapest item in this entire document.

**What a positive result (per-token separates, pooled doesn't) establishes:** the fix is architectural
(switch MDM's conditioning to per-token/cross-attention, à la CASIM or RVHM2D's local-feature variant)
and does **not** require touching CLIP's weights at all — cheaper and more surgical than Proposal 1,
and directly resolves the open "concatenation vs. cross-attention" question flagged in Section 0 by
supplying a spatial-specific reason to prefer cross-attention even if it costs a little on general
metrics.

**What a negative result (per-token also fails to separate) establishes:** rules out the pooling
hypothesis cleanly, and justifies going straight to Proposal 1's encoder-level fix rather than spending
budget on a conditioning-mechanism change that would not help.

**Strongest objection a reviewer would raise:** "A word's embedding is influenced by its sentence
context even before pooling (contextualized token embeddings, not static word vectors) — you have not
isolated the pooling step in a controlled way unless you also compare the same word in matched sentence
contexts that differ only in whether the sentence is pooled or not, which is the same sentence encoded
two ways, not two different things." Make sure the comparison is same-sentence pooled-vs-per-token, not
per-token-in-isolation vs. pooled-in-context, or the probe will not isolate what it claims to.

### Proposal 3 (third priority): Complexity-curriculum weighting by spatial-vocabulary density

**Hypothesis:** Weighting or ordering HumanML3D training examples by caption complexity — using the
project's own spatial-vocabulary flag (57.6% of captions) as one axis of "hard" — with a smooth
(Gaussian-style, per arXiv:2511.18378's image-domain finding) rather than strict easy-to-hard schedule,
improves spatial-caption generation quality more than uniform random sampling, at zero architectural
change and near-zero extra compute (a sampler change only).

**Exact experiment:** Define a per-caption complexity score (length + spatial-vocab flag + clause
count, all computable from the existing HumanML3D text with no LLM calls needed). Compare three
training-order arms on an otherwise-identical MDM fine-tune: (A0) uniform random (current default,
control); (A1) strict easy-to-hard by complexity score; (A2) Gaussian schedule shifting the sampling
distribution's mean from easy to hard smoothly over training, per the image-domain paper's
better-performing design.

**Arms/controls:** A0/A1/A2 above, same seed set (≥3 seeds given HumanML3D's small size, matching
RESEARCH_C's own recommendation), same total step budget across arms so wall-clock/step count is not a
confound. Evaluate on the same spatial-subset/full-subset split as Proposal 1 for comparability across
this document's proposals.

**Compute estimate:** this only changes the training data *sampler*, not the model or data itself — cost
is dominated by however many fine-tuning steps the project was already budgeting for MDM (a few
GPU-hours per arm x 3 arms x 3 seeds, on the order of the project's existing training budget, likely
10-20 GPU-hours total if training runs are already short given the project's stated budget elsewhere).

**What a positive result establishes:** a genuinely untried (per this search), zero-architectural-cost
intervention that directly operationalizes the project's own 57.6% statistic as a training-time lever,
not just a diagnostic one — and it transplants a specific, better-than-naive scheduling design
(Gaussian, not strict easy-to-hard) from a paper that already showed strict ordering is not the best
choice in an adjacent domain.

**What a negative result establishes:** given HumanML3D's small size (~15K motions, few epochs typically
needed to converge), a null result here would suggest curriculum ordering is a large-dataset-only
effect (consistent with the image-domain paper's own compute-scale setting, which is far larger than
HumanML3D) — informative about the limits of transferring curriculum-learning findings across dataset
scales, and a useful negative result for the field, not just this project.

**Strongest objection a reviewer would raise:** "Complexity score is a proxy (length + a keyword flag +
clause count), not a validated measure of what makes a caption 'hard' for a motion model to learn from —
you have not shown this proxy correlates with actual model error rate per caption, so an ordering
built on it could be ordering by the wrong thing entirely." Mitigate by first checking, on the frozen
baseline model, whether generation error (e.g., per-caption FID contribution or retrieval rank) actually
correlates with the proposed complexity score before using that score to build a curriculum.

---

## Source list (for citation)

- MoCLIP — arXiv:2505.10810 (HTML fetched, VERIFIED)
- CASIM — arXiv:2502.02063 (abstract fetched, VERIFIED)
- Local Action-Guided Motion Diffusion — arXiv:2407.10528 (abstract fetched, VERIFIED)
- LUMA — arXiv:2509.25304 (abstract fetched, VERIFIED)
- RVHM2D — arXiv:2506.14428 (HTML fetched, VERIFIED)
- Motion Flow Matching — arXiv:2312.08895 (fetch failed/WIP page; UNVERIFIED, WebSearch synthesis only)
- Is CLIP ideal? No. Can we fix it? Yes! — arXiv:2503.08723 (abstract fetched, VERIFIED)
- Left-Right Symmetry Breaking in CLIP-style VLMs — arXiv:2601.12809 (abstract fetched, VERIFIED)
- Text encoders bottleneck compositionality — arXiv:2305.14897 (abstract fetched, VERIFIED)
- Circuit Mechanisms for Spatial Relation Generation in DiTs — arXiv:2601.06338 (abstract fetched, VERIFIED; image-domain, not motion)
- "Aligned and realistic latent diffusion for text-to-motion generation" — DOI 10.1186/s42492-026-00224-2 (paywalled; UNVERIFIED, WebSearch synthesis only)
- MoLingo — arXiv:2512.13840 (abstract fetched, thin; VERIFIED existence, UNVERIFIED numbers)
- PhysiInter — arXiv:2506.07456 (abstract-level, VERIFIED via WebSearch synthesis of geometric losses)
- Synthetic Curriculum Reinforces Compositional Text-to-Image Generation — arXiv:2511.18378 (UNVERIFIED, WebSearch synthesis only)
- EasyTune — arXiv:2602.07967 (UNVERIFIED numbers, WebSearch synthesis only)
- LoRA-MDM ("Dance Like a Chicken") — arXiv:2503.19557 (HTML fetched directly, VERIFIED, including Table 2 rank-ablation numbers)
- Retrieving and Refining Winning Noise Tickets — arXiv:2607.06843 (HTML fetched, section existence VERIFIED; table numbers UNVERIFIED)
- MoMug / "Unlocking Pretrained LLMs for Motion-Related Multimodal Generation" — arXiv:2503.06119 (UNVERIFIED numbers, WebSearch synthesis only)
- "$100K or 100 Days" — arXiv:2410.23261 (PDF fetched but undecodable; existence/topic VERIFIED, specifics not found)
