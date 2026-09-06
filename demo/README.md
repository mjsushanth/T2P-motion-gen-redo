# Stage 5 demonstrator — caption truncation, made visible

Demonstrates this project's actual finding (`docs/EXPERIMENT_LOG.md`'s E1-pilot: caption
truncation costs measurable text-motion alignment, ~9-17x its own noise floor) rather than this
project's own model, which is severely undertrained (`docs/DECISIONS.md` D-24) and would
misrepresent the finding if shown as the point of the demo. Design rationale in full:
`reviews/REVIEW_QUEUE.md` SUP-20260906-43.

## What it shows

**Redesigned per review SUP-20260906-60/61/62** — the original design led with two *generated*
panes (full vs. truncated caption), which invited exactly the overclaim the interface's own
caveat forbade: that comparison was tested (E1B) and found statistically unresolvable at any
affordable sample size (`docs/DECISIONS.md` D-26). Showing it prominently, next to prose
explaining it can't be trusted, meant the most salient thing on the page contradicted the text
under it. The headline now visualises the effect that **is** measured, instead.

Type any motion caption. **Headline (instant, no generation, no GPU wait):**
1. Applies the **original failed course project's own caption-truncation rule** (faithfully
   reproduced, `demo/truncate.py` — same logic as `scripts/e1_pilot_caption_truncation.py`) live,
   so you see exactly what the original project's pipeline would have kept.
2. Retrieves the **nearest real, stored motion** for the full caption and for the truncated
   caption, separately — both panes are real HumanML3D motions, never generated, so generator
   quality is never a confound here. When they differ, that difference **is** the information
   the truncation rule destroyed, shown directly rather than asserted.
3. Prints the real, measured aggregate effect above the interaction: across 300 real captions,
   full captions find their own true motion 74.7% of the time, truncated captions 51.7% of the
   time — truncation changed the retrieved motion in 34.7% of cases
   (`artifacts/demo/self_retrieval_record.json`, `demo/measure_self_retrieval.py`).

**Secondary, below the fold, behind its own button:** generation via **MDM's own released
checkpoint** (MIT-licensed, not a model trained by this project — stated plainly, not implied as
ours), explicitly labelled illustrative, not evidential.

Nothing is curated. There is no example gallery — you type your own caption and can immediately
try something that breaks it. On roughly half of captions, truncation makes no visible
difference to retrieval; that null is part of the honest picture (one draw from a stated
distribution), not a broken demo.

## Requirements

- The `mjs_mlcvdl_unified_m5` conda environment (torch, numpy, scikit-learn already present),
  plus this directory's own pinned extras (`demo/requirements.txt`, per SUP-20260906-59):
  ```bash
  conda activate mjs_mlcvdl_unified_m5
  pip install -r demo/requirements.txt
  python -m spacy download en_core_web_sm   # separate download, not a pip dependency
  ```
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

- **The two *generated* panes' relative quality cannot be attributed to truncation** at this
  project's scale — that comparison (E1B) was run and found statistically unresolvable at any
  affordable sample size (`docs/DECISIONS.md` D-26). They are demoted below the fold and labelled
  illustrative-only for exactly this reason. The demo's actual evidence is the retrieval headline.
- **The retrieval baseline is TF-IDF, not the evaluator's own text-to-motion embedding space —
  checked, not assumed** (review SUP-20260906-58/62, `demo/measure_self_retrieval.py`): an
  embedding retriever was built first, on the reasoning that it uses the same validated space
  R-Precision itself relies on. Measured against TF-IDF on the identical 300-caption sample, its
  full-corpus self-retrieval rate collapsed to ~1% (TF-IDF: 74.7%) — diagnosed directly, not
  dismissed: a caption's own true motion scores 0.978 cosine similarity yet ranks 264th out of
  8,198 candidates, because the evaluator's text encoder was trained to discriminate a caption
  from ~31 decoys (R-Precision's own 32-candidate batch protocol), not to rank it first against
  an ~8,000-motion corpus — a real, disclosed limit on what that encoder generalizes to, not a
  reason to hide the result and quietly ship whichever retriever looked better. TF-IDF's
  74.7%/51.7% therefore measures a related but different quantity than R-Precision's
  0.8013/0.6563 (measured within 32-candidate batches) — both are real, both are reported,
  neither is claimed to be the other. The embedding retriever (`retrieval_embedding.py`) is kept
  in the codebase and available for a future rung where the corpus is small enough for it to
  matter, but is not currently used by `app.py`.
- **The demo's truncation was validated against the one actually measured, not assumed to match**
  (`validate_truncation_agreement.py`, review SUP-20260906-57): run over 8,962 real HumanML3D
  captions, the spaCy-based rule used here agrees with the tag-based rule the E1-pilot actually
  measured on **93.8%** of captions (punctuation-normalized comparison — the raw agreement rate,
  52.5%, is inflated apart mostly by whitespace/punctuation tokenization differences, not by
  where the rule decides to cut) and **98.1%** on which branch fired (conjunction-tag vs.
  first-sentence fallback). **The gap between those two figures (93.8% vs. 98.1%) means the same
  branch fires almost every time, but the exact cut point occasionally lands a token or two
  earlier or later than the tag-based rule's would** — so a reader should read this as
  reproducing the measured rule to ~94%, not exactly. Full record:
  `artifacts/demo/truncation_agreement_record.json`. The remaining disagreement comes from
  genuine POS-tagging differences between spaCy and HumanML3D's own tagger (e.g. hyphenated
  words, borderline ADP/SCONJ calls) — a real, small, disclosed gap, not zero.

## Files

- `app.py` — Gradio interface, orchestrates everything below.
- `truncate.py` — the original project's truncation rule, ported to run on free-form text via
  live POS-tagging (the E1-pilot's version needed HumanML3D's own pre-tagged captions only).
- `generate_wrapper.py` — thin wrapper around `sample/generate.py` (MDM's own script, reused
  directly, not reimplemented) for single-caption generation + rendering.
- `render_real_motion.py` — renders an already-stored real HumanML3D motion (for the retrieval
  pane), reusing MDM's own decode path (`recover_from_ric`) and renderer (`plot_3d_motion`).
- `retrieval.py` — the TF-IDF nearest-neighbour baseline; what `app.py` actually uses, per
  SUP-20260906-62's measurement (see Known limitations above).
- `retrieval_embedding.py` — text-to-motion embedding retrieval via the evaluator's own encoder;
  built to test whether it beats TF-IDF (per SUP-58), measured to not generalize to full-corpus
  retrieval at this scale, kept for future use rather than deleted.
- `measure_self_retrieval.py` — measures, over the same 300-caption sample, how often each
  retriever finds a caption's own true motion, full vs. truncated (SUP-20260906-60/61/62); this
  is the comparison that decided TF-IDF over the embedding retriever, and the source of the
  74.7%/51.7% numbers on the demo's headline.
- `validate_truncation_agreement.py` — measures how often `truncate.py`'s spaCy-based rule
  agrees with the tag-based rule the E1-pilot actually measured (SUP-20260906-57); result:
  93.8% punctuation-normalized agreement over 8,962 real captions.
- `_mpl_moviepy_compat.py` — a small compatibility shim (documented in its own docstring) for a
  real version mismatch between this environment's matplotlib and the old `moviepy` API MDM's
  vendored renderer expects; does not touch vendored code or downgrade shared dependencies.
