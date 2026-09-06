"""Stage 5 demonstrator (SUP-20260906-43, redesigned per SUP-20260906-60/61/62):
demonstrates the FINDING, not the model.

Headline (instant, no generation): type a caption, watch the original failed course project's
own truncation rule chop it, then see the nearest REAL motion each version retrieves. Both panes
are real, stored motions -- so generator quality is not a confound, and the difference between
them *is* the information truncation destroyed. This visualises the actual measured effect
(caption truncation costs retrieval accuracy) rather than an unresolvable generation comparison.

Retrieval uses TF-IDF, not the evaluator's own text-to-motion embedding space -- checked, not
assumed (review SUP-20260906-62, `demo/measure_self_retrieval.py`): the embedding retriever's
full-corpus self-retrieval collapses to ~1% (its encoder was trained to discriminate among
32 candidates, not ~8,000), while TF-IDF finds a caption's own true motion 74.7% of the time
(full) / 51.7% (truncated), independently re-measured against the director's own 78.3%/55.0%.
TF-IDF wins on stated, measured grounds, not because it looked better going in.

Secondary, below the fold, behind its own button: generation via MDM's own released checkpoint
(MIT-licensed, already verified in E0b) -- illustrative only, explicitly not evidential
(`docs/DECISIONS.md` D-26: the relative quality of the two generated panes cannot be attributed
to truncation at this project's affordable sample size).

Run: python app.py   (from this directory; needs the mjs_mlcvdl_unified_m5 conda env active)
"""
import os
import tempfile

import gradio as gr

from truncate import truncate_first_action_clause
from retrieval import NearestNeighborRetriever
from generate_wrapper import generate_video
from render_real_motion import render_real_motion

HEADLINE_MD = """
# Caption truncation: a finding, made visible

Type any motion caption below. Nothing here is curated — try something that breaks it.

**Measured across 300 real HumanML3D captions (`artifacts/demo/self_retrieval_record.json`,
TF-IDF retrieval, independently re-verified against the original measurement):**

| | full caption | truncated caption |
|---|---|---|
| finds its **own** true motion | **74.7%** of the time | **51.7%** of the time |

**Truncating the caption to the original failed course project's own rule changed which real
motion got retrieved in 34.7% of cases.** That is the same phenomenon as this project's
headline retrieval-space finding (R-Precision-top3 drops 0.8013 → 0.6563, ~9x the measured noise
floor, `docs/EXPERIMENT_LOG.md`'s E1-pilot) — shortening the description makes it stop finding
the right motion. **No embeddings, no metric jargon: type a caption, watch it happen (or not —
on any single caption, roughly half the time truncation makes no visible difference, and that
null is itself part of the honest picture, not a broken demo).**
"""

RETRIEVAL_DISCLAIMER_MD = """
Retrieval baseline is **TF-IDF** (lexical similarity), not the evaluator's own text-to-motion
embedding space — checked, not assumed: the embedding space was trained to discriminate a caption
from ~31 decoys (R-Precision's own protocol), not to rank it first against the ~8,000-motion
corpus this retrieval draws from, and it does not transfer to that harder task (self-retrieval
collapses to ~1%, `artifacts/demo/self_retrieval_record.json`). TF-IDF's 74.7%/51.7% measures a
related but different quantity than R-Precision's 0.8013/0.6563 — both are real, both are
reported, neither is claimed to be the other.
"""

GENERATION_MD = """
### Illustrative only — not evidential (`docs/DECISIONS.md` D-26)

Generation uses **MDM's own released checkpoint** (MIT license, GuyTevet/motion-diffusion-model)
— not a model trained by this project. This project's own model is severely undertrained
(3,000 steps, 0.63% of MDM's published budget) and would not represent this project's actual
finding fairly if shown here.

**The relative quality of the two panes below cannot be attributed to truncation** — that
comparison was tested directly (E1B) and found statistically unresolvable at any sample size
this hardware affords (0.80σ; resolving it would need ~1,780 generated samples per arm per
seed, ~9 CPU-hours each). The retrieval panes above are this demo's actual evidence; the panes
below just show what a real generator does with each caption. **Each generation takes several
minutes** (CPU-only, 1,000 diffusion steps, no GPU available).
"""


_TFIDF_RETRIEVER = None


def get_tfidf_retriever():
    global _TFIDF_RETRIEVER
    if _TFIDF_RETRIEVER is None:
        _TFIDF_RETRIEVER = NearestNeighborRetriever()
    return _TFIDF_RETRIEVER


def _truncate_with_note(caption: str):
    truncated_caption, used_fallback = truncate_first_action_clause(caption)
    fallback_note = (
        "(no conjunction found in this caption -- fell back to 'first sentence', "
        "per the original rule's own quirk, `docs/LANDMINES.md` §15)"
        if used_fallback else
        "(truncated at the first conjunction, matching the original project's own rule)"
    )
    return truncated_caption, fallback_note


def run_retrieval(caption: str):
    """Instant: no generation, no GPU wait. This is the demo's headline interaction."""
    caption = (caption or "").strip()
    if not caption:
        raise gr.Error("Type a caption first.")

    truncated_caption, fallback_note = _truncate_with_note(caption)
    retriever = get_tfidf_retriever()
    full_caption_match, full_id, full_sim = retriever.nearest(caption, k=1)[0]
    trunc_caption_match, trunc_id, trunc_sim = retriever.nearest(truncated_caption, k=1)[0]

    work_dir = tempfile.mkdtemp(prefix="t2p_demo_retrieval_")
    full_video = render_real_motion(full_id, os.path.join(work_dir, "full_retrieval.mp4"))
    trunc_video = render_real_motion(trunc_id, os.path.join(work_dir, "trunc_retrieval.mp4"))

    same_motion = full_id == trunc_id
    outcome_note = (
        "**Same real motion retrieved either way** — on this caption, truncation happened not "
        "to change the outcome (roughly half of captions land here; that is expected, not a "
        "failure of the demo)."
        if same_motion else
        "**A different real motion was retrieved** — truncating the caption changed what this "
        "system thinks the sentence describes."
    )
    truncation_summary = (
        f"**Full caption:** {caption}\n\n"
        f"**Truncated caption (the original project's own rule):** {truncated_caption}\n\n"
        f"{fallback_note}\n\n{outcome_note}"
    )
    match_summary = (
        f"**Full caption retrieved** (similarity {full_sim:.3f}): \"{full_caption_match}\" "
        f"(motion id: `{full_id}`)\n\n"
        f"**Truncated caption retrieved** (similarity {trunc_sim:.3f}): \"{trunc_caption_match}\" "
        f"(motion id: `{trunc_id}`)"
    )
    return full_video, trunc_video, truncation_summary, match_summary, truncated_caption


def run_generation(caption: str, truncated_caption: str, seed: int):
    caption = (caption or "").strip()
    if not caption:
        raise gr.Error("Run retrieval first (type a caption above).")
    work_dir = tempfile.mkdtemp(prefix="t2p_demo_gen_")
    full_video = generate_video(caption, os.path.join(work_dir, "full"), seed=seed)
    trunc_video = generate_video(truncated_caption or caption, os.path.join(work_dir, "truncated"), seed=seed)
    return full_video, trunc_video


with gr.Blocks(title="T2P-motion-gen-redo -- caption truncation demonstrator") as demo:
    gr.Markdown(HEADLINE_MD)

    caption_box = gr.Textbox(
        label="Your caption", placeholder="a person walks forward, swaying their arms",
    )
    retrieve_btn = gr.Button("Show what gets retrieved (instant)", variant="primary")

    truncated_caption_state = gr.State("")
    truncation_md = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Full caption → retrieved real motion")
            full_retrieval_out = gr.Video(label="Full caption retrieval")
        with gr.Column():
            gr.Markdown("### Truncated caption → retrieved real motion")
            trunc_retrieval_out = gr.Video(label="Truncated caption retrieval")
    match_md = gr.Markdown()
    gr.Markdown(RETRIEVAL_DISCLAIMER_MD)

    retrieve_btn.click(
        fn=run_retrieval,
        inputs=[caption_box],
        outputs=[full_retrieval_out, trunc_retrieval_out, truncation_md, match_md, truncated_caption_state],
    )

    gr.Markdown("---")
    gr.Markdown(GENERATION_MD)
    with gr.Row():
        seed_box = gr.Number(label="Seed", value=10, precision=0, scale=1)
        generate_btn = gr.Button("Also generate (several minutes)", scale=1)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Full caption → generated")
            full_gen_out = gr.Video(label="Full caption generation")
        with gr.Column():
            gr.Markdown("### Truncated caption → generated")
            trunc_gen_out = gr.Video(label="Truncated caption generation")

    generate_btn.click(
        fn=run_generation,
        inputs=[caption_box, truncated_caption_state, seed_box],
        outputs=[full_gen_out, trunc_gen_out],
    )

if __name__ == "__main__":
    demo.launch()
