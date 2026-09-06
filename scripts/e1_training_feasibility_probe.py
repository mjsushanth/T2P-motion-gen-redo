"""
E1 feasibility probe -- measures real per-training-step wall-clock cost of MDM's
vendored architecture on this machine, BEFORE any E1 arm is scaffolded or run.

Not a training run: no checkpoint is produced, no result is claimed. This only
answers "how long would training actually take," per the director's explicit
instruction to measure before committing (docs/DECISIONS.md D-23).

Reuses MDM's own train_args() parser (sys.argv construction, like the E0b driver
reused evaluation_parser()) so every hyperparameter is MDM's real default, not a
hand-built guess.
"""
import os
import sys
import time
import json
import argparse

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))

import torch  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils.parser_util import train_args  # noqa: E402
from data_loaders.get_data import get_dataset_loader  # noqa: E402
from utils.model_util import create_model_and_diffusion  # noqa: E402


def build_args(save_dir, batch_size, device_str):
    scratch_argv = [
        "probe",
        "--save_dir", save_dir,
        "--overwrite",
        "--dataset", "humanml",
        "--batch_size", str(batch_size),
        "--device", "0",
        "--diffusion_steps", "1000",
        "--arch", "trans_enc",
        "--layers", "8",
        "--latent_dim", "512",
        "--num_frames", "196",
    ]
    old_argv = sys.argv
    sys.argv = scratch_argv
    try:
        args = train_args()
    finally:
        sys.argv = old_argv
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--timed_steps", type=int, default=8)
    p.add_argument("--split", default="test", help="only 'test' is materialized on disk right now")
    cli = p.parse_args()

    scratch_save_dir = "/private/tmp/claude-501/-Users-joel-Library-Application-Support-Claude-scratch-workspaces-a646f0ea-d2d6-4531-a67f-9b65090d282c-02812373-138d-4a58-874b-33e46fa50871-scratch-2026-09-06-b94096/0c00fbc3-65fb-4192-9ac4-f079dfc7394a/scratchpad/e1_feasibility_probe_save_dir"
    os.makedirs(os.path.dirname(scratch_save_dir), exist_ok=True)

    fixseed(10)
    args = build_args(scratch_save_dir, cli.batch_size, cli.device)

    device_requested = cli.device
    device_used = device_requested
    if device_requested == "mps" and not torch.backends.mps.is_available():
        device_used = "cpu"

    print(f"requested device={device_requested}, using device={device_used}")
    print("loading data (split=%s, only split materialized on disk)..." % cli.split)
    data = get_dataset_loader(
        name="humanml", batch_size=cli.batch_size, num_frames=None,
        split=cli.split, hml_mode="train",
    )
    print(f"dataset size: {len(data.dataset)} sequences, batch_size={cli.batch_size}")

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    torch_device = torch.device(device_used)
    model.to(torch_device)
    n_params = sum(p.numel() for p in model.parameters_wo_clip())
    print(f"total trainable params (excluding CLIP): {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data_iter = iter(data)

    def next_batch():
        nonlocal data_iter
        try:
            motion, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(data)
            motion, cond = next(data_iter)
        motion = motion.to(torch_device)
        cond["y"] = {k: (v.to(torch_device) if torch.is_tensor(v) else v) for k, v in cond["y"].items()}
        return motion, cond

    from diffusion.resample import create_named_schedule_sampler
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    def run_one_step():
        motion, cond = next_batch()
        opt.zero_grad()
        t, weights = schedule_sampler.sample(motion.shape[0], device_used if device_used != "mps" else "cpu")
        t = t.to(torch_device)
        weights = weights.to(torch_device)
        losses = diffusion.training_losses(model, motion, t, model_kwargs=cond, dataset=data.dataset)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())

    print(f"warmup: {cli.warmup_steps} steps (not timed)...")
    for _ in range(cli.warmup_steps):
        run_one_step()

    print(f"timing {cli.timed_steps} real training steps...")
    step_times = []
    losses_seen = []
    for i in range(cli.timed_steps):
        t0 = time.perf_counter()
        loss_val = run_one_step()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        losses_seen.append(loss_val)
        print(f"  step {i}: {t1 - t0:.3f}s, loss={loss_val:.5f}")

    step_times_sorted = sorted(step_times)
    median_s = step_times_sorted[len(step_times_sorted) // 2]
    mean_s = sum(step_times) / len(step_times)

    record = {
        "device_requested": device_requested,
        "device_used": device_used,
        "batch_size": cli.batch_size,
        "split_used": cli.split,
        "dataset_size": len(data.dataset),
        "n_params_millions": n_params / 1e6,
        "warmup_steps": cli.warmup_steps,
        "timed_steps": cli.timed_steps,
        "step_times_seconds": step_times,
        "median_seconds_per_step": median_s,
        "mean_seconds_per_step": mean_s,
        "losses_seen": losses_seen,
        "note": "training-step timing only, no checkpoint produced, no result claimed",
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "e1", "e1_training_feasibility_probe_record.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"median s/step={median_s:.3f}, mean s/step={mean_s:.3f}, device={device_used}, params={n_params/1e6:.2f}M")


if __name__ == "__main__":
    main()
