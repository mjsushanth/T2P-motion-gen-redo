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
motion got retrieved in 34.7% of cases.** That is **the same underlying effect, measured a
different way**, as this project's headline retrieval-space finding (R-Precision-top3 drops
0.8013 → 0.6563, ~9x the measured noise floor, `docs/EXPERIMENT_LOG.md`'s E1-pilot) — lexical
retrieval and the validated embedding space are different mechanisms, but both show shortening
the description makes it stop finding the right motion. **No embeddings, no metric jargon: type
a caption, watch it happen (or not — on any single caption, roughly half the time truncation
makes no visible difference, and that null is itself part of the honest picture, not a broken
demo).**

*(A second, independent measurement of this same statistic — different sampling, different
corpus construction — found 78.3% → 55.0%, a 23.3-point drop, against this page's 23.0-point
drop: two runs agreeing to a third of a point on the number that matters. The absolute levels
differ by a few points because of denser vs. sparser corpus construction between the two runs —
noted for anyone who finds both numbers and wonders whether they measure the same thing. They do.)*
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
comparison was tested directly (E1B) and found a bounded null at this project's affordable
sample size (`docs/DECISIONS.md` D-28: effects ≥0.175 excluded at 3σ). The retrieval panes above
are this demo's actual evidence; the panes below just show what a real generator does with each
caption. **Each generation takes about 10 seconds** (MPS, `docs/DECISIONS.md` D-27/D-28 — this
was several minutes on CPU before MPS training/generation was validated; measured directly,
click to rendered video, not assumed from the batch-32 rate elsewhere in this project).
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
    # Show kept vs. discarded text directly (optional polish, per Review 13) -- the truncated
    # caption is a prefix of the full one by construction (truncate_first_action_clause only
    # ever removes a trailing clause), so the discarded remainder is just the suffix.
    if caption.lower().startswith(truncated_caption.lower().rstrip(".")):
        discarded = caption[len(truncated_caption):].strip(" ,.")
    else:
        discarded = None
    if discarded:
        cut_display = (
            f"**{truncated_caption}** ~~{discarded}~~\n\n"
            f"KEPT: \"{truncated_caption}\"  —  DISCARDED: \"{discarded}\""
        )
    else:
        cut_display = f"**Truncated caption (the original project's own rule):** {truncated_caption}"
    truncation_summary = (
        f"**Full caption:** {caption}\n\n"
        f"{cut_display}\n\n"
        f"{fallback_note}\n\n{outcome_note}"
    )
    # Similarity scores deliberately omitted (review SUP-20260906-68): TF-IDF cosine similarity
    # is not comparable between the full and truncated queries -- a shorter query mechanically
    # scores higher (fewer terms left unmatched in the query vector), independent of whether it
    # found the right motion. Displaying both numbers side by side invited exactly the wrong
    # reading (truncated query "matched better"), for a number that added nothing a non-
    # specialist needed -- which motion was retrieved is already legible from the captions/video.
    match_summary = (
        f"**Full caption retrieved:** \"{full_caption_match}\" (motion id: `{full_id}`)\n\n"
        f"**Truncated caption retrieved:** \"{trunc_caption_match}\" (motion id: `{trunc_id}`)"
    )
    return full_video, trunc_video, truncation_summary, match_summary


def run_generation(caption: str, seed: int):
    """Computes its own truncated caption (SUP-20260906-80): this button used to take
    truncated_caption from a gr.State that only run_retrieval ever populated, so clicking
    "Also generate" without first clicking "Show what gets retrieved" silently generated the
    SAME caption for both panels -- two identical videos under contrasting labels, the opposite
    of the finding this demo exists to show, with nothing on screen signalling it. Computing
    truncation here, the same way run_retrieval does, makes the two buttons unable to disagree
    and removes the click-order dependency entirely."""
    caption = (caption or "").strip()
    if not caption:
        raise gr.Error("Type a caption first.")
    truncated_caption, _ = truncate_first_action_clause(caption)
    work_dir = tempfile.mkdtemp(prefix="t2p_demo_gen_")
    full_video = generate_video(caption, os.path.join(work_dir, "full"), seed=seed)
    if truncated_caption == caption:
        # Truncation is a real no-op for some captions (no conjunction, nothing to cut) -- say so
        # loudly rather than silently rendering two panels that look like a contrast but aren't.
        trunc_video = full_video
        note = (
            "**Truncation was a no-op for this caption** (no conjunction found to cut) — both "
            "panels below show the identical generation, on purpose, not a bug."
        )
    else:
        trunc_video = generate_video(truncated_caption, os.path.join(work_dir, "truncated"), seed=seed)
        note = f"**Generated from:** full = \"{caption}\" · truncated = \"{truncated_caption}\""
    return full_video, trunc_video, note


with gr.Blocks(title="T2P-motion-gen-redo -- caption truncation demonstrator") as demo:
    gr.Markdown(HEADLINE_MD)

    caption_box = gr.Textbox(
        label="Your caption", placeholder="a person walks forward, swaying their arms",
    )
    retrieve_btn = gr.Button("Show what gets retrieved (instant)", variant="primary")

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
        outputs=[full_retrieval_out, trunc_retrieval_out, truncation_md, match_md],
    )

    gr.Markdown("---")
    gr.Markdown(GENERATION_MD)
    with gr.Row():
        seed_box = gr.Number(label="Seed", value=10, precision=0, scale=1)
        generate_btn = gr.Button("Also generate (~10s each, MPS)", scale=1)
    generation_note_md = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Full caption → generated")
            full_gen_out = gr.Video(label="Full caption generation")
        with gr.Column():
            gr.Markdown("### Truncated caption → generated")
            trunc_gen_out = gr.Video(label="Truncated caption generation")

    generate_btn.click(
        fn=run_generation,
        inputs=[caption_box, seed_box],
        outputs=[full_gen_out, trunc_gen_out, generation_note_md],
    )

if __name__ == "__main__":
    demo.launch()
