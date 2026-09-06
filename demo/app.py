"""Stage 5 demonstrator (SUP-20260906-43): demonstrates the FINDING, not the model.

Type a caption. Watch the original failed course project's own truncation rule chop it. See both
the full-caption and truncated-caption versions generate, side by side, plus a nearest-neighbour
retrieval baseline from real training data. The gap between the two generated panes is the
finding -- made visible to someone who will never read an R-Precision table.

Generation uses MDM's own released checkpoint (MIT-licensed, GuyTevet/motion-diffusion-model,
already verified in this project's E0b) -- NOT this project's own model, which is severely
undertrained (D-24: 0.63% of MDM's training budget) and would visibly misrepresent what this
project actually established. Quality here is real because the generator is real; this demo is
about CONDITIONING, and the generator is a stated, cited component, not implied as ours.

Run: python app.py   (from this directory; needs the mjs_mlcvdl_unified_m5 conda env active)
Each generation takes several minutes on this CPU-only hardware -- the UI says so plainly.
"""
import os
import tempfile

import gradio as gr

from truncate import truncate_first_action_clause
from retrieval import NearestNeighborRetriever
from generate_wrapper import generate_video
from render_real_motion import render_real_motion

METRICS_MD = """
### Measured, not asserted (E1-pilot, `docs/EXPERIMENT_LOG.md`)

| | full caption | truncated caption | drop |
|---|---|---|---|
| R-Precision-top3 (retrieval, real motions, n=4,648) | **0.8013** | **0.6563** | **0.145** (~9x the noise floor) |

This retrieval-space effect is real and resolved. Whether it **propagates into generated motion
quality** was tested directly (E1B) and found **not resolvable at any sample size this hardware
affords** (`docs/DECISIONS.md` D-26) -- the generated panes below show real output from a real
model, but the R-Precision *difference* between them, if any, cannot be claimed to be caused by
truncation at this scale. **Every number on this page is internally-comparable-only** — no claim
of comparability to published leaderboard results (`docs/DECISIONS.md` D-22).
"""

DISCLAIMER_MD = """
**Generation uses MDM's own released checkpoint** (MIT license, GuyTevet/motion-diffusion-model)
— not a model trained by this project. This project's own model is severely undertrained
(3,000 steps, 0.63% of MDM's published budget) and would not represent this project's actual
finding fairly if shown here. The finding this demo shows — that caption truncation destroys
measurable text-motion alignment — comes from the retrieval-space measurement above, not from
comparing generation quality between the two panes.

**Each generation takes several minutes** (CPU-only, 1,000 diffusion steps, no GPU available).
"""


def _load_retriever():
    return NearestNeighborRetriever()


_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = _load_retriever()
    return _RETRIEVER


def run_demo(caption: str, seed: int):
    caption = (caption or "").strip()
    if not caption:
        raise gr.Error("Type a caption first.")

    truncated_caption, used_fallback = truncate_first_action_clause(caption)
    fallback_note = (
        "(no conjunction found in this caption -- fell back to 'first sentence', "
        "per the original rule's own quirk, `docs/LANDMINES.md` §15)"
        if used_fallback else
        "(truncated at the first conjunction, matching the original project's own rule)"
    )

    retriever = get_retriever()
    nn_caption, nn_id, nn_sim = retriever.nearest(caption, k=1)[0]

    work_dir = tempfile.mkdtemp(prefix="t2p_demo_")
    full_video = generate_video(caption, os.path.join(work_dir, "full"), seed=seed)
    trunc_video = generate_video(truncated_caption, os.path.join(work_dir, "truncated"), seed=seed)
    nn_video = render_real_motion(nn_id, os.path.join(work_dir, "nn.mp4"))

    truncation_summary = (
        f"**Full caption:** {caption}\n\n"
        f"**Truncated caption (the original project's own rule):** {truncated_caption}\n\n"
        f"{fallback_note}"
    )
    retrieval_summary = (
        f"**Nearest real caption in the training corpus** (TF-IDF cosine similarity = "
        f"{nn_sim:.3f}):\n\n\"{nn_caption}\"\n\n(motion id: `{nn_id}`)"
    )
    return full_video, trunc_video, nn_video, truncation_summary, retrieval_summary


with gr.Blocks(title="T2P-motion-gen-redo -- caption truncation demonstrator") as demo:
    gr.Markdown("# Caption truncation: a finding, made visible")
    gr.Markdown(
        "Type any motion caption. Nothing here is curated -- try something that breaks it."
    )
    gr.Markdown(METRICS_MD)
    with gr.Row():
        caption_box = gr.Textbox(
            label="Your caption", placeholder="a person walks forward, swaying their arms",
            scale=4,
        )
        seed_box = gr.Number(label="Seed", value=10, precision=0, scale=1)
        run_btn = gr.Button("Generate (several minutes)", variant="primary", scale=1)
    gr.Markdown(DISCLAIMER_MD)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Full caption -> generation")
            full_out = gr.Video(label="Full caption")
        with gr.Column():
            gr.Markdown("### Truncated caption -> generation")
            trunc_out = gr.Video(label="Truncated caption")
        with gr.Column():
            gr.Markdown("### Nearest-neighbour retrieval (real motion, not generated)")
            nn_out = gr.Video(label="\"Just look it up\" baseline")
    truncation_md = gr.Markdown()
    retrieval_md = gr.Markdown()

    run_btn.click(
        fn=run_demo,
        inputs=[caption_box, seed_box],
        outputs=[full_out, trunc_out, nn_out, truncation_md, retrieval_md],
    )

if __name__ == "__main__":
    demo.launch()
