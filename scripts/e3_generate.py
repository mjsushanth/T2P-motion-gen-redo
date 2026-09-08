"""E3 generation pass -- generate every HumanML3D test-split caption once with MDM's released
checkpoint, no training, caching raw generated motions to disk one batch at a time so an
interrupted run resumes from the last completed batch instead of losing the whole ~3.7-hour run.

Mirrors CompMDMGeneratedDataset's own generation loop (data_loaders/humanml/motion_loaders/
comp_v6_model_dataset.py) exactly -- same sample_fn call, same batch construction, same
cap_len convention -- but iterates the loop directly instead of going through that class, so
each batch can be written to disk as soon as it's produced. Scoring (Guo + TMR R-Precision, FID,
spatial/non-spatial/tercile breakdown) happens in a separate pass over the cached batches
(scripts/e3_score.py) -- this script's only job is generation.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e3_generate.py --model-path <path> --out-dir <path> [--num-samples-limit N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-samples-limit", type=int, default=4198)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=10)
    my_args = ap.parse_args()

    out_dir = Path(my_args.out_dir)
    batches_dir = out_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Construct sys.argv for MDM's own evaluation_parser(), so args.json's real training
    # hyperparameters (diffusion_steps, guidance_param, arch, etc.) load from the checkpoint
    # itself -- not hand-reconstructed, same discipline as scripts/e0b_mdm_reproduction.py.
    sys.argv = [
        "e3_generate",
        "--model_path", my_args.model_path,
        "--eval_mode", "debug",
        "--dataset", "humanml",
    ]
    from utils.parser_util import evaluation_parser
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    from utils.sampler_util import ClassifierFreeSampleModel

    args = evaluation_parser()
    fixseed(my_args.seed)
    args.batch_size = my_args.batch_size

    print("Loaded args from checkpoint's args.json:")
    print(json.dumps({k: v for k, v in vars(args).items()
                       if k in ("diffusion_steps", "arch", "latent_dim", "layers",
                                "cond_mask_prob", "guidance_param", "dataset")}, indent=2))

    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print("Device:", device)

    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                     num_frames=None, split="test", hml_mode="eval")
    print(f"Eval-mode test-split loader ready: {len(gen_loader.dataset)} sequences, "
          f"batch_size={args.batch_size}, {len(gen_loader)} batches available")

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    print(f"Loading checkpoint from [{my_args.model_path}]...")
    load_saved_model(model, my_args.model_path, use_avg=args.use_ema)
    scale = args.guidance_param
    if scale != 1.:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()

    sample_fn = diffusion.p_sample_loop

    manifest_path = out_dir / "manifest.json"
    manifest = {"num_samples_limit": my_args.num_samples_limit, "batch_size": my_args.batch_size,
                "seed": my_args.seed, "guidance_param": scale,
                "diffusion_steps": args.diffusion_steps, "batches": []}
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))

    done_batch_idxs = {b["batch_idx"] for b in manifest["batches"]}
    n_done = sum(b["n"] for b in manifest["batches"])
    print(f"Resuming: {n_done}/{my_args.num_samples_limit} already cached "
          f"({len(done_batch_idxs)} batches)")

    t_start = time.time()
    with torch.no_grad():
        for i, (motion, model_kwargs) in enumerate(gen_loader):
            if n_done >= my_args.num_samples_limit:
                break
            if i in done_batch_idxs:
                continue

            model_kwargs['y'] = {k: v.to(device) if torch.is_tensor(v) else v
                                  for k, v in model_kwargs['y'].items()}
            motion = motion.to(device)
            tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
            if scale != 1.:
                model_kwargs['y']['scale'] = torch.ones(motion.shape[0], device=device) * scale

            t_b0 = time.time()
            sample = sample_fn(
                model, motion.shape, clip_denoised=False, model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                noise=None, const_noise=False,
            )
            t_b1 = time.time()

            bsz = motion.shape[0]
            motions_np = np.stack(
                [sample[bi].squeeze().permute(1, 0).cpu().numpy() for bi in range(bsz)]
            ).astype(np.float32)
            lengths_np = model_kwargs['y']['lengths'].cpu().numpy()
            captions = list(model_kwargs['y']['text'])
            # Same cap_len fix as CompMDMGeneratedDataset itself (index of eos/OTHER, not
            # len(tokens)) -- matches this project's own already-validated R-Precision numbers.
            cap_lens = [tokens[bi].index('eos/OTHER') + 1 for bi in range(bsz)]

            batch_path = batches_dir / f"batch_{i:04d}.npz"
            np.savez(batch_path, motions=motions_np, lengths=lengths_np)
            meta_path = batches_dir / f"batch_{i:04d}_meta.json"
            with open(meta_path, "w") as f:
                json.dump({"batch_idx": i, "n": int(bsz), "captions": captions,
                           "tokens": tokens, "cap_lens": cap_lens,
                           "batch_seconds": t_b1 - t_b0}, f)

            manifest["batches"].append({"batch_idx": i, "n": int(bsz)})
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            n_done += bsz
            elapsed = time.time() - t_start
            rate = elapsed / max(1, n_done - sum(
                b["n"] for b in manifest["batches"] if b["batch_idx"] in done_batch_idxs
            ))
            remaining = my_args.num_samples_limit - n_done
            eta_min = (rate * remaining / 60) if rate > 0 else float("nan")
            print(f"batch {i}: {bsz} samples in {t_b1-t_b0:.1f}s "
                  f"({n_done}/{my_args.num_samples_limit} total, eta {eta_min:.1f} min)",
                  flush=True)

    total_h = (time.time() - t_start) / 3600
    print(f"Generation pass done. {n_done} samples cached under {batches_dir} "
          f"({len(manifest['batches'])} batches). This run took {total_h:.2f}h.")


if __name__ == "__main__":
    main()
