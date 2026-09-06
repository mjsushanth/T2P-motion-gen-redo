# Stage 5 demonstrator — caption truncation, made visible

Demonstrates this project's actual finding (`docs/EXPERIMENT_LOG.md`'s E1-pilot: caption
truncation costs measurable text-motion alignment, ~9-17x its own noise floor) rather than this
project's own model, which is severely undertrained (`docs/DECISIONS.md` D-24) and would
misrepresent the finding if shown as the point of the demo. Design rationale in full:
`reviews/REVIEW_QUEUE.md` SUP-20260906-43.

## What it shows

Type any motion caption. The interface:
1. Applies the **original failed course project's own caption-truncation rule** (faithfully
   reproduced, `demo/truncate.py` — same logic as `scripts/e1_pilot_caption_truncation.py`) live,
   so you see exactly what the original project's pipeline would have kept.
2. Generates a motion from the **full** caption and from the **truncated** caption, side by side,
   using **MDM's own released checkpoint** (MIT-licensed, not a model trained by this project —
   stated plainly in the interface, not implied as ours).
3. Shows a **nearest-neighbour retrieval baseline** — the closest real caption/motion in the
   training corpus — so "why not just look it up" is answered directly rather than avoided.
4. Prints the actual measured numbers (0.8013 full vs. 0.6563 truncated R-Precision-top3, and
   the "internally-comparable-only" caveat) next to the panes, not just in a report nobody reads.

Nothing is curated. There is no example gallery — you type your own caption and can immediately
try something that breaks it.

## Requirements

- The `mjs_mlcvdl_unified_m5` conda environment (or an equivalent env with `torch`, `gradio`,
  `spacy` + `en_core_web_sm`, `scikit-learn`, `moviepy<2`, `matplotlib`).
- MDM's released checkpoint at `checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/
  model000475000.pt` (already fetched and verified in this repo's own E0b work — see
  `third_party/motion-diffusion-model/PATCHES.md` for provenance).
- HumanML3D's materialized subset at `third_party/motion-diffusion-model/dataset/HumanML3D/`
  (already materialized via `scripts/materialize_humanml3d_test_subset.py` for both `test` and
  `train` splits — the retrieval corpus reads every `.txt` file under that directory's `texts/`
  subfolder).

## Run it

```bash
conda activate mjs_mlcvdl_unified_m5
cd demo
python app.py
```

Opens on `http://127.0.0.1:7860`. **CPU-only, no GPU required** — each generation (1,000
diffusion steps, no respacing, same as every other generation this project has run) takes
several minutes. The interface says so; it is not a fast demo, and pretending otherwise would be
dishonest about what a laptop-scale, CPU-only reproduction can offer.

## Known limitations, stated rather than hidden

- **The two generated panes' *relative* quality cannot be attributed to truncation** at this
  project's scale — that comparison (E1B) was run and found statistically unresolvable at any
  affordable sample size (`docs/DECISIONS.md` D-26). The demo's finding is the retrieval-space
  measurement printed on screen, not a claim about which generated pane "looks better."
- The retrieval baseline is deliberately simple (TF-IDF cosine similarity over the materialized
  corpus's captions) — a legitimate, unglamorous baseline, not the strongest possible retrieval
  system, so a viewer can trust it was not tuned to make the demo look good.
- The POS-tagging step (`spacy`/`en_core_web_sm`) approximates HumanML3D's own tagging
  convention closely but not exactly (e.g. it does not lemmatize verbs) — this does not affect
  the truncation rule's logic (which only reads POS tags, not lemmas), but the truncated text
  shown may differ in minor surface form from how the original project's own pipeline would have
  tagged the identical sentence.

## Files

- `app.py` — Gradio interface, orchestrates everything below.
- `truncate.py` — the original project's truncation rule, ported to run on free-form text via
  live POS-tagging (the E1-pilot's version needed HumanML3D's own pre-tagged captions only).
- `generate_wrapper.py` — thin wrapper around `sample/generate.py` (MDM's own script, reused
  directly, not reimplemented) for single-caption generation + rendering.
- `render_real_motion.py` — renders an already-stored real HumanML3D motion (for the retrieval
  pane), reusing MDM's own decode path (`recover_from_ric`) and renderer (`plot_3d_motion`).
- `retrieval.py` — the nearest-neighbour baseline.
- `_mpl_moviepy_compat.py` — a small compatibility shim (documented in its own docstring) for a
  real version mismatch between this environment's matplotlib and the old `moviepy` API MDM's
  vendored renderer expects; does not touch vendored code or downgrade shared dependencies.
