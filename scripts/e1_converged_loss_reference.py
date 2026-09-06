"""Converged-loss reference, per review SUP-20260906-35.

Diffusion training loss has no reference scale on its own -- it averages over uniformly
sampled timesteps, so most of its batch-to-batch variance is which timesteps got drawn, not
model quality. Watching E1A's raw loss trace for 3,000 steps cannot say whether that budget is
"enough" without something to compare it against.

This script computes that reference: run MDM's actual released checkpoint (model000475000.pt,
475,000 real training steps, the number our ~3,000-step budget is 0.63% of) through the SAME
training_losses() call, on the SAME data loader and batch distribution E1A trains on, for a
fixed number of batches. Forward passes only -- no backward, no generation, minutes not hours.

The seed is fixed identically before both this reference computation and E1A's own training loop
(same fixseed() call, same loader construction order), so both draw the same batch/timestep
sequence as far as is achievable without literally sharing process state -- removing as much of
the "which timesteps got drawn" noise from the comparison as this setup allows.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1_converged_loss_reference.py --model-path <path> --out-json <path>
"""
import argparse
import json
import os
import sys
from pathlib import Path

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--out-json", required=True)
    my_args = ap.parse_args()

    sys.argv = [
        "e1_converged_loss_reference",
        "--model_path", my_args.model_path,
        "--eval_mode", "debug",
        "--dataset", "humanml",
    ]
    from utils.parser_util import evaluation_parser
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    from diffusion.resample import create_named_schedule_sampler
    import torch

    args = evaluation_parser()
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print(f"device: {device}")

    fixseed(my_args.seed)
    print("building the SAME loader/batch distribution E1A trains on "
          "(split=test, hml_mode=train, batch_size matched)...")
    data = get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                               num_frames=None, split="test", hml_mode="train")

    print(f"loading released checkpoint from [{my_args.model_path}]...")
    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, my_args.model_path, use_avg=args.use_ema)
    model.to(device)
    model.eval()

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # Reset the seed immediately before the measured loop, so the shuffle order and the
    # timestep draws below match what E1A's own training loop draws when it is fixseed()'d
    # the same way immediately before its own loop starts.
    fixseed(my_args.seed)
    data_iter = iter(data)
    losses = []
    with torch.no_grad():
        for i in range(my_args.num_batches):
            try:
                motion, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(data)
                motion, cond = next(data_iter)
            motion = motion.to(device)
            cond["y"] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond["y"].items()}
            t, weights = schedule_sampler.sample(motion.shape[0], device)
            loss_terms = diffusion.training_losses(model, motion, t, model_kwargs=cond, dataset=data.dataset)
            loss = (loss_terms["loss"] * weights).mean()
            losses.append(float(loss))
            if i % 10 == 0:
                print(f"  batch {i}/{my_args.num_batches}: loss={loss:.5f}")

    losses_sorted = sorted(losses)
    mean_loss = sum(losses) / len(losses)
    median_loss = losses_sorted[len(losses_sorted) // 2]

    result = {
        "experiment": "E1-converged-loss-reference",
        "description": "Forward-only training_losses() on MDM's released 475,000-step "
                       "checkpoint, same loader/batch distribution as E1A, as a reference "
                       "scale for E1A's own training loss trace.",
        "model_path": my_args.model_path,
        "num_batches": my_args.num_batches,
        "batch_size": my_args.batch_size,
        "seed": my_args.seed,
        "losses": losses,
        "mean_loss": mean_loss,
        "median_loss": median_loss,
        "note": "eval() mode (dropout off) on the reference; E1A's own training loop computes "
                "loss in train() mode (dropout on), so a small, expected, structural gap remains "
                "even at full convergence -- this is a reference scale, not an exact bound.",
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
