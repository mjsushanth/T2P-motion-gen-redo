# Patches to vendored motion-diffusion-model (MDM)

Cloned verbatim from `github.com/GuyTevet/motion-diffusion-model` (default branch, 2026-09-06),
MIT licensed (verified via GitHub API). One file patched; documented here per the same
discipline as `primary_source/PATCHES.md` and `../text-to-motion/PATCHES.md`.

## model/rotation2xyz.py

`Rotation2xyz.__init__` unconditionally instantiated a real SMPL body model
(`self.smpl_model = SMPL().eval().to(device)`), which requires `body_models/smpl/SMPL_NEUTRAL.pkl`
-- a gated asset requiring account registration at smpl.is.tue.mpg.de (SMPL's own licence,
`LANDSCAPE.md` §4.1). This project's non-commercial research/education use is explicitly
permitted by that licence (D-20), but registering an account is a step only Joel can take
personally, not something this session does on his behalf.

**This submodule is not needed for this project's use case.** `MDM`'s constructor always builds
`Rotation2xyz` regardless of dataset, but it is only ever exercised when converting *rotation*-
based pose representations to xyz joint positions (`a2m`/`humanact12`/`uestc`, or optional mesh
visualization). For `dataset='humanml'` with `data_rep='hml_vec'` (this project's use, and the
representation behind every published HumanML3D FID/R-Precision number), `Rotation2xyz.__call__`
returns early (`if pose_rep == "xyz": return x`) before `self.smpl_model` is ever touched.

**Patch:** made `self.smpl_model` lazy — set to `None` in `__init__`, constructed on first actual
use inside `__call__` (immediately before the line that needs it). This does not change behavior
for any caller that *does* have the SMPL files and needs the rotation-to-xyz path; it only removes
the hard, unconditional dependency for callers (like this project's E0b) that never reach it.

## model/mdm.py

Same reason as above: `MDM._apply` and `MDM.train` both unconditionally touch
`self.rot2xyz.smpl_model` (needed so `.to(device)`/`.train()` propagate to the SMPL submodule
when it exists). Guarded both with `if self.rot2xyz.smpl_model is not None:` to match the lazy
load in `rotation2xyz.py`. No effect on behavior when `smpl_model` is actually constructed.

## DataLoader `num_workers` (multiple files)

Set every `num_workers=1/4/8` to `num_workers=0` in `data_loaders/get_data.py`,
`data_loaders/humanml/motion_loaders/{dataset_motion_loader,comp_v6_model_dataset,
model_motion_loaders}.py`. This code was developed/tested on Linux, where PyTorch's default
multiprocessing start method is `fork` (worker processes inherit the parent's memory, so a local
`lambda` collate function works fine). macOS's default start method is `spawn` (worker processes
are freshly started and need everything pickled), and `data_loaders/get_data.py`'s
`get_collate_fn` returns a `lambda` in every branch — `AttributeError: Can't pickle local object
'get_collate_fn.<locals>.<lambda>'` on macOS with any `num_workers > 0`. Pure performance
parameter (single-process vs multi-process data loading); no effect on correctness or on any
number this project reports. `data_loaders/humanml/networks/trainers.py`'s `num_workers=4` (used
only for *training* a new evaluator from scratch, never exercised by this project) left
untouched.

## Reproducing E0b's local (gitignored) inputs

Not committed, per the upstream repo's own `.gitignore` convention (`t2m/`, `glove/`) plus a
project-added rule for the materialized subset (`dataset/HumanML3D/`, `dataset/t2m_test.npy`):

```bash
# 1. Evaluator checkpoint (same one E0a uses) -- symlink, do not duplicate the 245MB file:
mkdir -p t2m/text_mot_match/model
ln -sf ../../../../checkpoints/t2m/text_mot_match/model/finest.tar t2m/text_mot_match/model/finest.tar

# 2. GloVe word vectors (already vendored under ../text-to-motion/glove/):
mkdir -p glove
cp ../text-to-motion/glove/*.npy ../text-to-motion/glove/*.pkl glove/

# 3. Materialized HumanML3D test-split subset:
cd ../.. && python3 scripts/materialize_humanml3d_test_subset.py \
  --n-samples 128 --out-dir third_party/motion-diffusion-model/dataset/HumanML3D \
  --mean-std-src checkpoints/t2m_dataset_stats
```

The MDM checkpoint itself (`checkpoints/mdm/humanml-encoder-512/`, 413MB, gitignored) is fetched
via `gdown` from the official Google Drive link in this project's upstream MDM README.

No other files in this clone have been modified.
