# Verification note on RESEARCH_A / RESEARCH_B / RESEARCH_C

**Written by the reviewing architect, 2026-09-07T00:35Z. Read this BEFORE acting on anything in
those three files.**

The three research documents in this directory were produced by delegated research agents. I
independently re-fetched the load-bearing citations rather than accepting them. **The pattern that
emerged: the agents are reliable on direction and unreliable on specific statistics.** Two of the
numbers most likely to drive an experimental design turned out to be wrong. Treat every unchecked
number in those files as UNVERIFIED regardless of how it is labelled there.

## Verified by me, directly from the source

| claim | source | status |
| :-- | :-- | :-- |
| TMR reduces median retrieval rank 54 -> 19 | arXiv:2305.00976 | **CONFIRMED** — title, authors (Petrovich/Black/Varol), and number all match the abstract |
| VideoMDM trains a 3D motion prior from 2D video only; FID 0.88 vs fully-supervised 0.54 | arXiv:2606.13364 | **CONFIRMED** — and its 0.54 baseline matches MDM's published FID in `LANDSCAPE.md`, an independent consistency check |
| Rectified flow beats diffusion head; **~13 hours on a single RTX 5090**; FID 0.192 (flow, epoch 54) vs 0.240 (diffusion, epoch 142); R@3 0.828 vs 0.807 | arXiv:2603.26747 | **CONFIRMED** from the paper's HTML. Not in the abstract — the abstract alone does not support it. Exact quote: "each of the diffusion and rectified flow models requires approximately 13 hours of training on a single NVIDIA RTX 5090 GPU." |
| Contrastive VLM text encoders fail on left/right positional relations | arXiv:2311.11477 | **CONFIRMED in direction.** Title is "What's left can't be right -- the remaining positional incompetence of contrastive vision-language models." |

## Corrected — do not propagate these

| claim as written in the research files | what is actually true |
| :-- | :-- |
| "BLIP scored 56% vs 99% human on spatial minimal pairs" attributed to arXiv:2311.11477 | **Not in that paper's abstract.** The qualitative finding stands; this statistic is attached to the wrong source. Do not cite it. |
| "HumanML3D++ (arXiv:2404.14745): the standard Guo evaluator misjudged semantically-correct pairs **~40% of the time**, forcing the authors to abandon R-Precision/FID" | **Does not hold.** The paper is real — "You Think, You ACT: The New Task of Arbitrary Text to Motion Generation" — but contains no such percentage. The nearest statement is from a 10% sample: "we find that 66% of the data can be similar to the ground truth," which is about *LLM-generated action text quality*, not evaluator failure. The abstract does support the weaker claim that the authors used "multi-solution metrics to address the inadequacies of existing single-solution metrics." **The 40% figure appears to be a derivation from 66% reframed as evaluator error. It must not be cited, and no experimental design may rest on it.** |

## Resolved: an apparent conflict that was not one

`LANDSCAPE.md` §1.4 describes arXiv:2411.16575 as a paper proposing "a more robust evaluation
method." RESEARCH_A describes the same ID as MARDM, a masked-autoregressive diffusion model with
FID 0.114. **Both are correct.** The paper is "Rethinking Diffusion for Text-Driven Human Motion
Generation: Redundant Representations, Evaluation, and Masked Autoregression" — it contributes
*both* the architecture and the evaluation critique. This makes it the single most relevant paper
to this project, not a citation error.

## Still unverified by me (agent-reported only)

MRBench (arXiv:2608.07993) 63.2% caption-collision / 148-motions / median 11 words · arXiv:2309.10248
metric-vs-human correlation 0.27-0.82 · SnapMoGen caption-augmentation ablation FID 17.98->15.56 ·
arXiv:2310.16656 recaptioning FID 17.87->14.84 · FAST DCT tokenization (arXiv:2501.09747) ·
Motion Mamba training cost · all LoRA-for-motion citations. **Verify before use.** The MRBench
caption-collision numbers in particular would be load-bearing for any evaluator critique, and are
exactly the class of statistic that was wrong twice above.
