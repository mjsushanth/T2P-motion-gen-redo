"""Thin wrapper around MDM's own sample/generate.py -- reuses its main() directly rather than
reimplementing generation+rendering, per this project's standing rule (wrap vendored machinery,
do not reimplement it). Produces one rendered .mp4 per call, using MDM's released checkpoint
(MIT-licensed, already verified in E0b) -- generation quality is real because the model is real;
this demo is about conditioning, not about our own (severely undertrained, 0.63% of MDM's
training budget) model.

Device: whatever `dist_util.dev()` resolves to -- CPU or MPS, not chosen here (SUP-20260906-80).
D-24 ("MPS is unusable") is REVERSED by D-27/D-28: the same `_extract_into_tensor` and
`evaluator_wrapper.py` patches that make E1A/E1B trustworthy on MPS apply here unchanged, since
this wrapper calls the same vendored generation code. Full 1000 diffusion timesteps (no
respacing), consistent with every other generation call this project has made (E0b, E1A, E1B) --
not a novel, unverified speedup path, just the same call on whichever device `dist_util` picks.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mpl_moviepy_compat  # noqa: E402,F401 -- applies the tostring_rgb shim on import

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "mdm", "humanml-encoder-512",
    "humanml_trans_enc_512", "model000475000.pt",
)


def generate_video(caption: str, out_dir: str, seed: int = 10) -> str:
    """Generates one motion sample for `caption` using MDM's released checkpoint, renders it to
    an .mp4 via MDM's own plot_3d_motion, and returns the path to that file.

    Must be called with cwd set to MDM_ROOT (its own relative-import structure requires this,
    same constraint as every other script in scripts/ that reuses vendored MDM code)."""
    abs_out_dir = os.path.abspath(out_dir)
    abs_checkpoint = os.path.abspath(CHECKPOINT)
    if os.path.exists(abs_out_dir):
        shutil.rmtree(abs_out_dir)

    scratch_argv = [
        "generate",
        "--model_path", abs_checkpoint,
        "--text_prompt", caption,
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--output_dir", abs_out_dir,
        "--seed", str(seed),
    ]
    old_argv = sys.argv
    old_cwd = os.getcwd()
    sys.path.insert(0, os.path.abspath(MDM_ROOT))
    try:
        os.chdir(MDM_ROOT)
        sys.argv = scratch_argv
        from utils.parser_util import generate_args
        import sample.generate as gen_module

        args = generate_args()
        gen_module.main(args)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    # sample/generate.py's own save_multiple_samples() writes the real file here, using its
    # "all" naming template -- the per-sample "sample00_rep00.mp4" name it also constructs is
    # only ever used as an argument to plot_3d_motion(), which does not write to that path itself.
    video_path = os.path.join(abs_out_dir, "samples_00_to_00.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"expected rendered video at {video_path}, not found")
    return video_path
