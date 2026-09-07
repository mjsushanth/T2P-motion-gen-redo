# Research C: Data-Centric Approaches — Caption Augmentation for Text-to-Motion

**Question under test:** "Captions are the bottleneck, not the architecture" — for HumanML3D-scale
text-to-motion training (~15k motions, ~12-word captions), does LLM caption augmentation move the
needle, and can it be tested cheaply and honestly?

**Epistemic key:** VERIFIED = I fetched the source directly (arXiv abs/html page, or a page whose
content I inspected) and am reporting what it contains. UNVERIFIED = surfaced only by a web search
snippet/secondary aggregator that I did not independently fetch and confirm. No number below appears
unless it came from a source in one of these two categories, and the category is stated every time.

---

## 1. LLM caption/prompt augmentation as a training intervention (cross-domain precedent)

### DALL-E 3 (Betker et al., OpenAI, 2023) — the canonical precedent
Source: `cdn.openai.com/papers/dall-e-3.pdf` (primary PDF — **fetch failed**, exceeded the 10MB
content-length limit of my fetch tool). Reported instead via a secondary source that reproduces the
paper's claims and figures: https://shreyansh26.github.io/post/2024-02-18_dalle3_image_recaptioner/
— **UNVERIFIED (secondary summary of a source I could not directly open)**.

- OpenAI trained a dedicated image captioner (fine-tuned to describe subject, surroundings, spatial
  layout, visible text, style, color) and used it to re-caption the entire training corpus before
  training the generator.
- Final training mix: **95% synthetic (model-generated) captions, 5% ground-truth captions.** They
  report this ratio maximizes model performance on their internal CLIP-score-based evaluation, but
  the exact CLIP score deltas per mixing ratio were not recoverable from the secondary source (charts
  only, not tables).
- **Distribution-shift risk they identified and named explicitly:** models trained predominantly on
  long, syntactically regular synthetic captions "overfit to the text distribution" — picking up
  synthetic-captioner artifacts (letter casing, punctuation placement, caption length/style) rather
  than genuine semantics.
- **Their mitigation (directly relevant to this project's Risk #4):** (a) during training, randomly
  select ground-truth vs. synthetic caption per example at a fixed probability (the 95/5 mix) rather
  than using 100% synthetic, specifically to keep some grounding in the real distribution; (b) at
  **inference time**, use GPT-4 to "upsample" short user prompts into detailed descriptions so that
  what the model sees at inference matches what it saw in training. This second point is the closest
  documented analogue to "evaluator/train-test caption mismatch" in the literature: OpenAI's answer
  was not to leave inference-time text alone, but to rewrite it to match the training distribution.
- No hallucination-rate number was recoverable from the secondary source.

### Sora and Wan (text-to-video) — same recipe, explicit train/inference alignment framing
Source: WebSearch synthesis over `openai.com/index/video-generation-models-as-world-simulators`,
survey papers, and the Wan technical report (arXiv:2503.20314) — **UNVERIFIED (not independently
fetched)**.
- Sora explicitly re-applies the DALL-E 3 recaptioning recipe to video: trains a descriptive video
  captioner, recaptions the full training set, and reports the resulting captions "improve text
  fidelity as well as the overall quality of videos" (no numeric delta recovered).
- Wan's technical report reportedly names the same two-sided fix as DALL-E 3: augment each training
  video with multiple diverse captions, **and** rewrite user prompts at inference to match the
  training caption distribution. This is presented as a distinct, necessary step — not a side effect
  of just having better captions.
- No arXiv ID exists for the original DALL-E 3/Sora reports (Sora has no paper, only a technical
  blog); Wan's arXiv ID is 2503.20314 but I did not fetch it directly, so treat all Wan specifics as
  UNVERIFIED pending direct confirmation.

### PixArt-α and "A Picture is Worth a Thousand Words" — the numbers I could actually verify
- **PixArt-α** (arXiv:2310.00426) — **VERIFIED (fetched abs page)**, but the abs page only confirmed
  the qualitative claim (LLaVA auto-labels dense captions from the SAM dataset as Stage-2 training
  data; "concept density" in captions is called out as the key lever) and the overall training-cost
  number (10.8% of SD1.5's compute), which is a whole-system number, not isolated to captioning.
  Caption-specific ablation numbers were **not found** in the abs page.
- **"A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation"**
  (arXiv:2310.16656) — **VERIFIED (fetched abs page directly)**. This is the cleanest isolated
  ablation I found anywhere in this research pass:
  - FID: **14.84 (recaptioned) vs. 17.87 (original captions)** — lower is better, ~17% relative
    improvement from recaptioning alone.
  - Human evaluation: **64.3%** relative improvement in "faithful image generation" (recaptioned
    preferred).
  - Semantic object accuracy: **84.34 vs. 78.90**.
  - Counting-alignment error: **1.32 vs. 1.44** (lower is better).
  - Positional alignment: **62.42 vs. 57.60**.
  - Dataset size for this ablation was **not found** in the abs-page content I could fetch.
  - The paper's own framing: recaptioning "reduces the train-inference discrepancy and provides the
    model with more information per example" — i.e., they attribute part of the gain specifically to
    closing a train/inference gap, not just to adding detail.

**Bottom line for #1:** the DALL-E-3-style recipe (recaption train set with a dedicated captioner,
mix with some ground truth, and upsample inference-time prompts to match) is well-established across
image and video generation, with at least one directly-verified isolated ablation (arXiv:2310.16656)
showing a real, non-trivial FID and human-preference gain from recaptioning alone. Cost data (compute
or $ for recaptioning) was not recoverable from any source I could fetch — mark as **not found**.

---

## 2. Synthetic/augmented captions specifically for motion (HumanML3D / KIT-ML)

Three distinct interventions exist in the literature; they are not the same thing, and conflating
them would be a mistake for this project's experimental design.

### (a) SnapMoGen (arXiv:2507.09122) — **VERIFIED**, fetched full HTML text
- Dataset: 20K motion clips, 44 hours, 122K descriptions averaging **48 words** vs. HumanML3D's 12.
- Each clip gets **6 descriptions: 2 human-authored + 4 LLM-augmented** (LLM appears to be ChatGPT,
  based on the prompt tables; exact model version not stated in the fetched text).
- **Direct ablation isolating caption augmentation from architecture (their Table 5):**
  - Without text augmentation: **FID 17.98, CLIP Score 0.656**
  - With text augmentation: **FID 15.56, CLIP Score 0.684**
  - This is a ~13% relative FID improvement and the paper states plainly: "caption augmentation
    clearly improves model performance across all evaluation metrics." This ablation holds the
    architecture (MoMask++) fixed, so the gain is attributable to the caption intervention.
  - **Critical caveat for this project:** this ablation is evaluated on SnapMoGen's *own* benchmark/
    evaluator, which was itself built around SnapMoGen's expressive-caption distribution. It does
    **not** show what happens if you augment captions but keep evaluating with the original, frozen
    Guo-et-al. HumanML3D text-motion co-embedding (see Section 4 — this is precisely the
    evaluator-contamination scenario this project needs to worry about, and SnapMoGen's result does
    not speak to it).
  - No discussion of caption hallucination risk was found in the fetched text — the paper's
    limitations section discusses mocap artifacts, not LLM-introduced content not grounded in motion.
  - Inference-time pipeline: casual user prompts are rewritten (~60 words) to add physical-movement
    detail before being fed to the model — the same train/inference-alignment logic as DALL-E 3/Sora.

### (b) HumanML3D++ (from "You Think, You ACT: The New Task of Arbitrary Text to Motion Generation,"
arXiv:2404.14745) — **VERIFIED**, fetched full HTML text. This is the single most load-bearing
finding for this project's Risk #4.
- Method: LLaMA fine-tuned via LoRA generates "scene texts" — plausible antecedent/contextual
  sentences that do **not** restate the action verb — via a "causal context-guided prompt" (few-shot,
  verb-restricted, causally-framed).
- Quality control: 20 human raters manually checked 15% of generated scene texts; generation was
  iterated **until validation accuracy exceeded 95%** (three-evaluator agreement required for a
  scene text to pass).
- Final size: **135K scene texts + 45K action texts over the same 15K motions** (~195K text-motion
  pairs), i.e. roughly the same base motions as HumanML3D but ~4x the caption volume via LLM
  generation.
- **The critical, directly-relevant finding:** when they ran the *existing* HumanML3D text-motion
  co-embedding extractor (the standard Guo-et-al. evaluator R-precision/FID is built on) against the
  new scene texts, it showed **a misjudgment rate approaching 40%**, "despite identical outputs to
  ground truth" — i.e., the standard evaluator's embedding space, fit on the original ~12-word action
  caption distribution, could not reliably score motion-caption pairs drawn from a different
  (LLM-generated, differently-distributed) caption style, even when the underlying semantic content
  was correct. This forced the authors to abandon R-Precision/FID for scene texts and invent new
  metrics (Hit Accuracy, Mean Hit Distance) instead.
  - Reported numbers on their own new metrics: Scene texts scored Hit Accuracy 79.9% vs. Action
    texts' R-Precision 0.696 (metrics are not directly comparable — different scales/definitions —
    so do not read this as "scene texts perform better"; it is reported here only to show the paper
    replaced the metric rather than reusing it unmodified).
- **This is the strongest piece of direct, verified evidence for this project's Risk #4** (evaluator
  contamination / distribution shift between augmented captions and the frozen evaluator). It is
  motion-specific, it is quantified (≈40% misjudgment rate), and it is exactly the failure mode the
  project is worried about — not a hypothetical.

### (c) Fg-T2M++ (arXiv:2502.05534) — **UNVERIFIED**, PDF fetch returned binary/unparseable content;
what follows is from WebSearch-surfaced secondary summaries (themoonlight.io review, Springer IJCV
listing, Semantic Scholar) and should be treated as low-confidence pending direct verification.
- **Important distinction:** Fg-T2M++'s "LLM augmentation" is an LLM-based *semantic parsing module
  at inference/training time* that extracts body-part-level descriptions and syntactic dependency
  structure from the *existing* caption — it is not DALL-E-3-style recaptioning of the training
  corpus with a richer, longer synthetic caption. Do not cite this as evidence for "richer captions
  help" without noting this is a different mechanism (structured parsing, not caption enrichment).
  reported deltas — FID reportedly dropping from 0.571 (Fg-T2M) to 0.135 (Fg-T2M++) — are UNVERIFIED
  and additionally confounded by simultaneous architecture changes (hyperbolic text representation
  module, multi-modal fusion module), so this cannot be read as an isolated caption-augmentation
  effect even if the numbers are accurate.

**Bottom line for #2:** motion-specific caption augmentation has been tried and shown to help
(SnapMoGen, isolated ablation, own evaluator) — but the one paper that actually stress-tested the
*standard* HumanML3D evaluator against LLM-augmented/differently-distributed captions
(arXiv:2404.14745) found it breaks (≈40% misjudgment), which is a serious, verified warning sign for
any experiment plan that (a) augments training captions but (b) reports results on the stock
Guo-et-al. R-Precision/FID pipeline without addressing this.

---

## 3. Data scaling vs. model scaling for small datasets

### "On the Scalability of Diffusion-based Text-to-Image Generation" (arXiv:2404.02883) —
**VERIFIED**, fetched abs page directly.
- Tested models from 0.4B–4B parameters against datasets up to 600M images.
- Direct quote: **"the quality and diversity of the training set matters more than simply dataset
  size"**; **"increasing caption density and diversity improves text-image alignment performance and
  the learning efficiency."**
- On the model side: increasing transformer blocks is more parameter-efficient for alignment than
  increasing channel width; they found a UNet variant 45% smaller and 28% faster than SDXL's UNet at
  comparable quality.
- **Gap: the abstract does not give a quantitative crossover point** (a data scale or model scale at
  which one factor stops mattering relative to the other). This is a genuine "not found" — I could
  not locate a source with an explicit crossover threshold, in this domain or in motion generation.

### Motion-specific scaling evidence — weak/absent
- "Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data" (arXiv:2507.07095) —
  **VERIFIED (abs page fetched)**: introduces MotionMillion (2M sequences, 2000+ hours) and scales
  the model to 7B params, but the abstract contains **no ablation isolating data scale from model
  scale**, and **no discussion of caption quality/length** as a separate variable. Not usable as
  evidence either way.
- "HY-Motion 1.0" (arXiv:2512.23464) — **VERIFIED (abs page fetched)**: scales a DiT flow-matching
  model to billion-parameter scale on 3,000+ hours of data with "meticulous... captioning," but
  again **no explicit data-vs-model bottleneck analysis** is present in the abstract.
- "The Quest for Generalizable Motion Generation: Data, Model, and Evaluation" (ViMoGen,
  arXiv:2510.26794) — **VERIFIED (fetched full HTML)**. Closest thing to a relevant ablation: their
  Table 6 shows swapping the text encoder from CLIP to T5-XXL improves their Motion Condition
  Consistency score from **0.35 → 0.44**, and they state "more powerful text encoders are essential
  for handling the greater linguistic complexity required for true generalization" — this is *some*
  evidence that encoder/model capacity matters for exploiting richer text, but it does not by itself
  separate "caption quality was the bottleneck" from "model capacity was the bottleneck," since both
  were changed together in spirit (better encoder = more capacity to use richer captions). They also
  report, without a number I could extract, that **"training with descriptive video-style text while
  testing on concise motion-style descriptions yields the best performance"** — an intriguing,
  verified claim that train-rich/test-terse caption mismatch was not harmful in their setup, which
  cuts against the naive version of the distribution-shift worry, but this is on their own MBench
  evaluator, not the standard frozen HumanML3D evaluator, so it does not resolve Section 4's concern.

**Bottom line for #3:** the text-to-image literature has a clear, verified qualitative finding
("quality/diversity of captions matters more than raw dataset size," arXiv:2404.02883) but no
quantitative crossover point. The motion literature has **no verified paper that isolates caption
quality from model capacity as the binding constraint** at HumanML3D's scale (~15k motions). This is
a real gap — the "captions are the bottleneck" hypothesis is well-precedented by analogy (image/video
domain) and directly argued qualitatively by SnapMoGen's dataset paper, but I found no motion paper
that runs the controlled experiment (same architecture, same data, caption-quality-only variable) at
HumanML3D scale. That gap is exactly what this project's proposed experiment (Section 6) would fill.

---

## 4. Risks and failure modes

This section is the most important one per your brief; I am being explicit about what is verified,
what is analogical, and what remains genuinely unfound.

1. **Caption augmentation introducing hallucinated content not grounded in the motion.**
   - **Not found**: no paper I could fetch (SnapMoGen, HumanML3D++, Fg-T2M++, or the T2I precedents)
     explicitly measures or reports a hallucination rate for LLM-augmented motion captions. SnapMoGen
     mitigates this partly by construction — the augmentation prompt explicitly instructs the LLM to
     preserve "the described body movements" exactly while only improving readability/grammar (a
     narrow rewrite, not free elaboration) — but they do not report a hallucination audit.
   - HumanML3D++'s scene-text pipeline built in an explicit anti-hallucination control (95% agreement
     among 3 evaluators required before a generated text is accepted), which is the strongest
     verified mitigation pattern found, but it targets contextual-plausibility, not motion-grounding
     per se (scene texts are deliberately *not* supposed to restate motion content).

2. **Distribution shift between augmented training captions and clean evaluation captions.**
   - **Verified precedent that this is a real, named risk**: DALL-E 3 (secondary source) explicitly
     identifies synthetic-caption overfitting (casing, punctuation, length/style artifacts) as a
     failure mode and mitigates with a 95/5 train mix plus inference-time prompt upsampling.
   - **Verified motion-specific evidence that this bites hard**: HumanML3D++'s ≈40% misjudgment rate
     when the standard evaluator is fed differently-distributed captions (arXiv:2404.14745) — this is
     the concrete, quantified version of exactly this risk.
   - **Not found**: a paper that trains a motion model with augmented captions and then reports
     side-by-side R-Precision/FID *on the original short-caption test split* vs. an *augmented-caption
     test split*, to directly show the metric's sensitivity to this shift. This specific controlled
     comparison appears not to exist yet in the literature I could access — treat as an open
     experimental question, not a settled one.

3. **Evaluator contamination** (if the evaluator was trained on original captions, does augmenting
   training text break comparability?).
   - This is functionally the same failure mode as #2, viewed from the evaluator's side, and the
     HumanML3D++ finding (Section 2b) is the best available verified evidence: the standard Guo-et-al.
     text-motion co-embedding, when scoring text drawn from a caption distribution it wasn't fit on,
     misjudges semantically-correct pairs at a high rate (~40%).
   - Separately, "What is the Best Automated Metric for Text to Motion Generation?"
     (arXiv:2309.10248) — **VERIFIED, fetched full HTML** — independently critiques the standard
     metrics on other grounds: R-Precision correlates only 0.816 with human "Faithfulness" ratings at
     the model level and is "poorly suited for sample-level comparisons"; FID correlates only 0.714
     with Faithfulness and a weak 0.269 with "Naturalness"; Multimodal Distance is "consistently weak"
     and they "recommend against" using it. They explicitly flag that these metrics "rely on a text
     and motion co-encoder, so proving the effectiveness of the encoder is crucial" — a direct,
     verified acknowledgment that the encoder (and by extension, the caption distribution it was fit
     on) is a load-bearing, under-scrutinized assumption of the whole evaluation pipeline. They propose
     a learned alternative (MoBERT, sample-level correlation with human judgment 0.624 vs. prior best
     0.208) but MoBERT's own robustness to caption-distribution shift is **not addressed** in what I
     fetched — another open gap.
   - **Direct implication for this project**: any experiment that augments HumanML3D training captions
     and then reports R-Precision/FID improvements using the stock Guo-et-al. evaluator is at risk of
     reporting an artifact of evaluator/caption-distribution mismatch rather than a genuine motion-
     quality gain (in either direction — it could inflate or deflate the measured effect, and the
     literature gives no reason to assume a particular sign). This must be controlled explicitly (see
     Section 6).

4. Cost/quality trade-off of the recaptioning model itself — **not found** for motion. No source
   fetched reports $ or GPU-hours spent on the captioner/augmentation step for any of DALL-E 3, Sora,
   SnapMoGen, or HumanML3D++. Budget for this experimentally rather than assuming a number.

---

## 5. Cheap open LLMs for local caption rewriting on Apple Silicon

All throughput figures below are **UNVERIFIED** — surfaced via WebSearch synthesis of benchmark blogs
(modelpiper.com, sitepoint.com, hybrid-llm.com, kunalganglani.com, llmcheck.net, ollama.com/blog/mlx),
not independently fetched and cross-checked against raw benchmark data, with one partial exception
(modelpiper.com fetched directly, see below).

- **modelpiper.com — VERIFIED (fetched directly)**: Llama 3.1 8B, Q4_K_M quantization, 2K context:
  - M2 Air 16GB (estimated): ~18 tok/s
  - M2 Pro 32GB (estimated): ~22 tok/s
  - M2 Max 32GB (measured): ~28 tok/s
  - Generation speed only; prompt processing runs 2-3x faster than generation. Context length matters:
    expect ~15-20% slowdown at 8K context vs. 2K. Only Llama 3.1 8B and Qwen2.5 14B were covered in
    this specific source — no 7B Qwen/Mistral numbers were present.
- **UNVERIFIED, from search synthesis**: Mac M3 (128GB config), Llama 3 8B ≈ 58 tok/s via Ollama.
  Ollama shipped an MLX backend in preview (dated to "0.19," reportedly March 2026 per one source)
  that on larger MoE models (Qwen3-Coder-30B-A3B) reportedly jumps from ~43 tok/s (old llama.cpp
  backend) to ~130 tok/s (MLX backend) on an M4 Pro Mac mini — a ~3x gain from the backend switch
  alone, unrelated to model choice. MLX vs. llama.cpp/Ollama-default is reported to lead by roughly
  15-30% throughput and ~10% less memory for models under 14B, converging above ~27B where memory
  bandwidth dominates regardless of runtime.
- **Practical framing for this project** (my calculation, not a cited number): at a conservative
  ~20-30 tok/s generation on a mid-tier M2/M3 (16-32GB) running an 8B-class model (Llama 3.1 8B,
  Qwen2.5 7B/8B — both fit comfortably in RAM at Q4 quantization), and assuming a rewritten caption
  runs ~40-80 output tokens, rewriting all ~45K HumanML3D captions once (single augmented version per
  caption) is on the order of 45,000 x 60 tokens / 25 tok/s ≈ 30 hours of wall-clock generation time,
  single-threaded, on a laptop. Producing SnapMoGen-style 4x augmentation per motion (4 rewrites x
  ~14.6K motions, using existing captions as seeds) is roughly the same order of magnitude. This is
  the kind of experiment that comfortably fits in an overnight or weekend local run, well within
  "cheap" — no cloud API spend required, though a cloud API (GPT-4o-mini class) would be faster and
  is worth costing out separately if local throughput becomes the bottleneck to iteration speed. No
  API costs were verified/cited here since the ask was specifically for local/Ollama/MLX options.
- Model recommendation for the caption-rewriting task specifically (my synthesis, not sourced):
  Llama-3.1-8B-Instruct or Qwen2.5-7B/14B-Instruct are the standard choices repeatedly named across
  the fetched/searched benchmark sources for this class of hardware; both are small enough to run at
  usable throughput on a 16-32GB Apple Silicon laptop via Ollama or MLX, and both are instruction-
  tuned models well-suited to a "rewrite this caption, preserve the described actions exactly" prompt
  in the style SnapMoGen used (Table 6 of arXiv:2507.09122, verified above).

---

## 6. Proposed experimental design: caption augmentation on HumanML3D, ~25K-100K training steps

**Hypothesis under test:** H1 — replacing/supplementing HumanML3D's ~12-word captions with LLM-
rewritten, more descriptive captions improves text-to-motion generation quality, holding architecture
and compute budget fixed, by more than can be explained by evaluator/distribution-shift artifacts.

### Arms (all trained from the same code, same architecture, same seed set, same step budget)
- **A0 — Baseline.** Original HumanML3D captions, unmodified. This is the control.
- **A1 — Augmented-only.** Every training caption replaced by one LLM rewrite (SnapMoGen-style:
  preserve described actions exactly, add fluent detail/readability, target a fixed longer word
  count, e.g. ~40-50 words to mirror SnapMoGen's own target).
- **A2 — Augmented + original mix.** Following DALL-E 3's finding that 100% synthetic overfits to
  synthetic-text artifacts: sample original vs. augmented caption per example at a fixed ratio each
  epoch (start at the DALL-E-3-reported 95/5 synthetic/original as a prior, but treat this ratio
  itself as a hyperparameter to sweep at small scale before committing, since DALL-E 3's ratio was
  tuned for image captions of a different length/failure-mode profile).
- **A3 — Length-matched control (critical, and easy to skip by accident).** Rewrite captions to the
  same *word count* as A1 but WITHOUT adding new semantic content — e.g., paraphrase/pad with
  filler/repetition, or use a weaker LLM prompted only to "make this longer" without instruction to
  add correct detail. This arm exists to separate "more semantic detail helped" from "the model just
  does better with longer input sequences / more text tokens to attend to," which is a confound the
  image-generation literature does not appear to have controlled for either (none of the sources
  above report a length-matched-nonsense control) — this project would be doing something the
  literature has not yet done, which is exactly the kind of arm worth pre-registering.

### Evaluation protocol (this is where most of the rigor needs to go, per Section 4's findings)
1. **Report on the frozen, standard Guo-et-al. evaluator (R-Precision/FID/MM-Dist) using the
   ORIGINAL, unmodified HumanML3D test-split captions for every arm**, including A1/A2/A3 whose
   *training* captions differed. This isolates: did the augmented-caption model get better at the
   task the field actually benchmarks on, using the eval distribution everyone else uses? This is the
   number that is comparable to prior published work.
2. **Separately, report the same metrics using an augmented version of the test captions** (rewritten
   the same way as the training augmentation, arm-matched). Comparing (1) vs. (2) for each arm
   directly measures the evaluator-distribution-sensitivity risk quantified in arXiv:2404.14745 (their
   ≈40% misjudgment figure) for your specific setup, rather than assuming it transfers from their
   scene-text case to your caption-rewrite case.
3. **Sanity-check the evaluator itself before trusting either number**: run the frozen Guo-et-al. text
   encoder on a held-out sample of {original caption, augmented caption} pairs known to describe the
   *same* motion, and check retrieval/embedding-similarity between the two caption embeddings. If
   augmented and original captions for the same motion do not land close together in the evaluator's
   embedding space, you have direct, local evidence (not borrowed from another paper) that the
   evaluator cannot be trusted at face value for the augmented arms, and metric (1) above needs a
   qualifying footnote or a supplementary metric.
4. **Human/qualitative check on a fixed sample (e.g., 50-100 generations) across all arms**, blind to
   arm identity, rated for text-motion faithfulness — because arXiv:2309.10248's verified finding is
   that R-Precision/FID/MM-Dist correlate only moderately (0.7-0.8 at best, and R-Precision is not
   even usable sample-level) with human faithfulness judgments in the first place, independent of any
   caption-augmentation question. Do not rely on automated metrics alone regardless of which arm wins.
5. **Hallucination audit on the augmented captions themselves**, independent of model training: sample
   ~200 augmented captions, have a human (or a second, different LLM as a cheap first pass, flagged
   for human spot-check) verify the added detail is plausibly consistent with the original short
   caption and not fabricated content contradicting it. Report the rate. This closes the "not found"
   gap in Section 4.1 for your own data, since no prior paper reports this number.

### Confounds to control explicitly (checklist)
- **Compute/step budget must be identical across arms** — more input tokens per example (longer
  captions) can change effective batch composition/memory and wall-clock steps-per-hour even at fixed
  gradient-step count; report both step count and wall-clock/GPU-hours per arm.
- **Text encoder/tokenizer must be identical and must not be re-fit per arm** — if the text encoder is
  trainable, freeze it across arms or you conflate "better captions" with "encoder adapted to a new
  text distribution."
- **Random seed and data order**: same seeds, same shuffling scheme across arms, multiple seeds per
  arm (at minimum 3) given HumanML3D's small size (~15K motions) makes single-seed results noisy.
- **The augmentation LLM and prompt must be fixed and logged verbatim** (model name/version, exact
  system+user prompt, temperature/sampling settings) so the caption-generation step itself is
  reproducible and auditable for the hallucination check above.
- **A3's length-matched-nonsense control is not optional** — without it, any win for A1/A2 over A0 is
  equally consistent with "the model benefits from longer text inputs regardless of content" as with
  "captions were semantically the bottleneck," and the whole point of this project is to distinguish
  those.
- **Report training-caption/test-caption distributional statistics** (average length, vocabulary
  size/type-token ratio, verb diversity) for every arm so a reviewer can see exactly how far training
  and eval distributions diverged, rather than asserting they didn't.
- **Pre-register the primary metric and the decision rule before running arms** (e.g., "H1 is
  supported if A1 or A2 beats A0 on frozen-evaluator/original-test-caption R-Precision Top-3 by more
  than X points AND the length-matched control A3 does not show a comparable gain AND the hallucination
  audit rate on augmented captions is below Y%") — given this project's stated aversion to fooling
  itself, the decision rule should be written down before the runs, not fit to whichever arm happens
  to look best afterward.

### What ~25K-100K steps buys you here
This step range is small enough that HumanML3D-scale training (15K motions) will complete multiple
epochs many times over per arm on typical hardware (exact wall-clock depends on architecture — this
project's own architecture choice, not something the sources above specify) — so it is well-suited to
running all 4 arms x 3 seeds = 12 runs within a modest compute budget, which is what the multi-seed
recommendation above assumes is affordable at this scale. If compute is tighter than that, cut arm
count (A3 is the most defensible to keep if only 2 arms are affordable, since it directly tests the
single biggest confound) before cutting seed count.

---

## Source list (for citation)

- DALL-E 3 (Betker et al.) — `cdn.openai.com/papers/dall-e-3.pdf` (fetch failed, size limit);
  secondary summary at https://shreyansh26.github.io/post/2024-02-18_dalle3_image_recaptioner/
- Sora — https://openai.com/index/video-generation-models-as-world-simulators/ (not fetched, UNVERIFIED)
- Wan — arXiv:2503.20314 (not fetched, UNVERIFIED)
- PixArt-α — arXiv:2310.00426 (abs page fetched)
- "A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation" —
  arXiv:2310.16656 (abs page fetched, VERIFIED numbers)
- SnapMoGen — arXiv:2507.09122 (full HTML fetched, VERIFIED)
- HumanML3D++ / "You Think, You ACT" — arXiv:2404.14745 (full HTML fetched, VERIFIED)
- Fg-T2M++ — arXiv:2502.05534 (PDF fetch failed/unparseable; UNVERIFIED secondary summaries only)
- "On the Scalability of Diffusion-based Text-to-Image Generation" — arXiv:2404.02883 (abs page
  fetched, VERIFIED)
- "Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data" — arXiv:2507.07095 (abs
  page fetched, VERIFIED but no usable ablation)
- "HY-Motion 1.0" — arXiv:2512.23464 (abs page fetched, VERIFIED but no usable ablation)
- "The Quest for Generalizable Motion Generation: Data, Model, and Evaluation" (ViMoGen) —
  arXiv:2510.26794 (full HTML fetched, VERIFIED)
- "What is the Best Automated Metric for Text to Motion Generation?" — arXiv:2309.10248 (full HTML
  fetched, VERIFIED)
- Apple Silicon LLM throughput — https://modelpiper.com/blog/local-llm-benchmarks-apple-silicon
  (fetched, VERIFIED for the specific numbers attributed to it); other throughput figures (Ollama MLX
  backend, M3/M4 estimates) are UNVERIFIED, WebSearch-surfaced only.
