"""Materialize a subset of the HumanML3D test split on disk, in the exact directory layout
MDM's data loader (data_loaders/humanml/data/dataset.py) expects, so its unmodified code can
run without adapting it to HF-streaming (unlike E0a, where adapting was the right call for a
much simpler pipeline). Source: HF `TeoGchx/HumanML3D`, streamed, not bulk-downloaded.

Writes:
    <out_dir>/new_joint_vecs/<id>.npy   -- raw 263-d motion array
    <out_dir>/texts/<id>.txt            -- caption line(s), format matching the official
                                            HumanML3D text file convention
    <out_dir>/test.txt                  -- list of ids (this subset only)
    <out_dir>/Mean.npy, Std.npy          -- official HumanML3D normalization stats (copied)
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset

MIN_MOTION_LEN = 40
MAX_MOTION_LEN = 199


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--mean-std-src", type=str, required=True,
                         help="dir containing Mean.npy/Std.npy to copy in")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (out_dir / "texts").mkdir(parents=True, exist_ok=True)

    shutil.copy(Path(args.mean_std_src) / "Mean.npy", out_dir / "Mean.npy")
    shutil.copy(Path(args.mean_std_src) / "Std.npy", out_dir / "Std.npy")

    print(f"Streaming HumanML3D test split (target {args.n_samples} samples)...")
    ds = load_dataset("TeoGchx/HumanML3D", split="test", streaming=True)
    ids = []
    n_seen = 0
    for i, ex in enumerate(ds):
        n_seen += 1
        motion = np.asarray(ex["motion"], dtype=np.float32)
        if motion.shape[0] < MIN_MOTION_LEN or motion.shape[0] > MAX_MOTION_LEN or motion.shape[1] != 263:
            continue
        # The raw caption field is itself multiple newline-separated entries, each ALREADY
        # in the official "<caption>#<tokens>#<from_tag>#<to_tag>" format (confirmed by direct
        # inspection, not assumed) -- earlier version of this script did caption.split("#") on
        # the whole multi-line string, which corrupted every entry after the first (verified via
        # a real E0b run: it silently dropped every sample because MDM's own dataset loader
        # wraps per-sample processing in a bare try/except that swallows the resulting parse
        # error). Fix: pass each already-correct line through unchanged.
        caption_field = ex["caption"].strip()
        lines = [ln for ln in caption_field.split("\n") if ln.strip() and "#" in ln]
        if not lines:
            continue
        sample_id = f"sample{i:06d}"
        np.save(out_dir / "new_joint_vecs" / f"{sample_id}.npy", motion)
        with open(out_dir / "texts" / f"{sample_id}.txt", "w") as f:
            for ln in lines:
                f.write(ln.strip() + "\n")
        ids.append(sample_id)
        if len(ids) >= args.n_samples:
            break

    with open(out_dir / "test.txt", "w") as f:
        for sid in ids:
            f.write(sid + "\n")

    print(f"Seen {n_seen} rows -> materialized {len(ids)} samples to {out_dir}")


if __name__ == "__main__":
    main()
