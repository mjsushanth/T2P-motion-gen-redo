# Vendored: EricGuo5513/text-to-motion (the D-03 evaluation harness)

Fetched verbatim from `github.com/EricGuo5513/text-to-motion` (default branch, 2026-09-06).
This is the actual evaluator behind every published HumanML3D number in `LANDSCAPE.md` §1.3 —
`EvaluatorModelWrapper` (text/motion encoders trained via `train_tex_mot_match.py`) plus
`final_evaluations.py`, `utils/metrics.py`, `motion_loaders/`.

## Licence (VERIFIED 2026-09-06)

MIT — confirmed via GitHub API (`GET /repos/EricGuo5513/text-to-motion/license` returns
`license.key: "mit"`) and by fetching the actual `LICENSE` file content directly (not inferred
from the key alone). Copyright (c) 2022 Chuan Guo. Verbatim text saved at `LICENSE` in this
directory.

## Checkpoint provenance (`checkpoints/t2m/text_mot_match/model/finest.tar`, gitignored, not in this repo's git history)

The official checkpoint is distributed via a Google Drive zip link in the upstream README, which
cannot be fetched headlessly without additional tooling. Instead, downloaded
`model/finest.tar` (245,580,211 bytes) from a third-party Hugging Face re-upload,
`Tevior/text_mot_match` (Apache-2.0), whose filename and directory structure
(`model/finest.tar`, `eval/E005.txt`...`E065.txt`) exactly matches what
`train_tex_mot_match.py` produces.

**Provenance is NOT independently verified beyond architecture match.** Confirmed, this session:
loaded the checkpoint's state dicts (`text_encoder`, `motion_encoder`, `movement_encoder`) and
checked every tensor shape against what `networks/evaluator_wrapper.py`'s `build_models()` and
`networks/modules.py`'s `MovementConvEncoder`/`TextEncoderBiGRUCo`/`MotionEncoderBiGRUCo`
constructors expect for the `t2m` (HumanML3D) config (`dim_pose=263`, `dim_word=300`,
`dim_motion_hidden=1024`, `dim_text_hidden=512`, `dim_coemb_hidden=512`, `POS_enumerator` of size
15). Every shape matched exactly (e.g. `movement_encoder.main.0.weight` is `(512, 259, 4)`,
matching `Conv1d(dim_pose-4=259, 512, 4)`). This is strong runtime evidence the checkpoint is
architecturally identical to what the original evaluator expects, but it is not a cryptographic
or author-confirmed match to the specific Google-Drive-hosted file — label this checkpoint's
exact provenance UNVERIFIED beyond the architecture-match evidence above.

## No other modifications

All vendored `.py` files are unmodified from what was fetched. `glove/our_vab_data.npy`,
`our_vab_idx.pkl`, `our_vab_words.pkl` are the repo's own committed GloVe word-vector files,
fetched verbatim (not gitignored, small enough to vendor directly).
