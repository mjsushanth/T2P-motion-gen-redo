"""Time MDM generation (the E1 cost-dominant term) on CPU vs MPS.

Not a quality check: the model is freshly initialized, never trained. Generation wall-clock
time depends on architecture, batch size, and diffusion step count, not on the model's weights,
so an untrained model gives an honest speed measurement without spending training time on it.
Mirrors the exact generation call in scripts/e1_train_arm.py (get_mdm_loader, batch_size=32,
num_samples_limit=128, guidance_param=2.5, 1000 diffusion steps) so the result is directly
comparable to E1A/E1B's own recorded generation_wall_clock_seconds.

Run from third_party/motion-diffusion-model/, env mjs_mlcvdl_unified_m5:
    E1_DEVICE=cpu python3 ../../scripts/e1_generation_timing_probe.py
    E1_DEVICE=mps python3 ../../scripts/e1_generation_timing_probe.py
"""
import json
import os
import sys
import time
from pathlib import Path

SEED = 10
BATCH_SIZE = 32
NUM_SAMPLES_LIMIT = int(os.environ.get("E1_NUM_SAMPLES_LIMIT", "128"))
DEVICE_REQUEST = os.environ.get("E1_DEVICE", "cpu")

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts" / "e1" / f"e1_generation_timing_probe_{DEVICE_REQUEST}_n{NUM_SAMPLES_LIMIT}.json"
)


def main():
    scratch_argv = [
        "e1_generation_timing_probe",
        "--save_dir", "/tmp/e1_gen_timing_probe_save_dir",
        "--overwrite",
        "--dataset", "humanml",
        "--batch_size", str(BATCH_SIZE),
        "--device", "0" if DEVICE_REQUEST != "cpu" else "-1",
        "--seed", str(SEED),
        "--diffusion_steps", "1000",
        "--arch", "trans_enc",
        "--layers", "8",
        "--latent_dim", "512",
        "--num_frames", "196",
    ]
    old_argv = sys.argv
    sys.argv = scratch_argv
    try:
        from utils.parser_util import train_args
        args = train_args()
    finally:
        sys.argv = old_argv

    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    from utils.model_util import create_model_and_diffusion
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
    import torch

    fixseed(SEED)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print(f"requested={DEVICE_REQUEST}, resolved device={device}")
    if DEVICE_REQUEST == "mps" and device.type != "mps":
        raise RuntimeError(f"requested mps but resolved to {device} -- dist_util.dev() patch not in effect")
    if DEVICE_REQUEST == "cpu" and device.type != "cpu":
        raise RuntimeError(f"requested cpu but resolved to {device}")

    print("loading eval-split data (ground truth + eval loaders, no training data needed)...")
    train_data = get_dataset_loader(name="humanml", batch_size=BATCH_SIZE, num_frames=None,
                                     split="test", hml_mode="train")

    print("creating model and diffusion (freshly initialized, untrained -- timing only)...")
    model, diffusion = create_model_and_diffusion(args, train_data)
    model.to(device)
    model.eval()

    guidance_param = 2.5
    sample_model = ClassifierFreeSampleModel(model) if guidance_param != 1 else model
    sample_model.to(device)

    gt_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                    split="test", hml_mode="gt")
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                     split="test", hml_mode="eval")

    if device.type == "mps":
        torch.mps.synchronize()
    t_gen_start = time.perf_counter()
    motion_loader, mm_motion_loader = get_mdm_loader(
        args, model=sample_model, diffusion=diffusion, batch_size=32,
        ground_truth_loader=gen_loader, mm_num_samples=0, mm_num_repeats=0,
        max_motion_length=gt_loader.dataset.opt.max_motion_length,
        num_samples_limit=NUM_SAMPLES_LIMIT, scale=guidance_param,
    )
    if device.type == "mps":
        torch.mps.synchronize()
    t_gen_end = time.perf_counter()
    generation_wall_clock_s = t_gen_end - t_gen_start
    n_batches = -(-NUM_SAMPLES_LIMIT // BATCH_SIZE)
    print(f"generation done: {generation_wall_clock_s:.1f}s for {NUM_SAMPLES_LIMIT} samples "
          f"({n_batches} batches of {BATCH_SIZE}), {generation_wall_clock_s / n_batches:.1f}s/batch, "
          f"{generation_wall_clock_s / NUM_SAMPLES_LIMIT:.2f}s/sample")

    result = {
        "experiment": "generation timing probe (CPU vs MPS)",
        "device_requested": DEVICE_REQUEST,
        "device_resolved": str(device),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_samples_limit": NUM_SAMPLES_LIMIT,
        "diffusion_steps": 1000,
        "guidance_param": guidance_param,
        "generation_wall_clock_seconds": generation_wall_clock_s,
        "seconds_per_batch": generation_wall_clock_s / n_batches,
        "seconds_per_sample": generation_wall_clock_s / NUM_SAMPLES_LIMIT,
        "note": "freshly initialized untrained model -- timing only, not a quality measurement. "
                "Mirrors scripts/e1_train_arm.py's own generation call exactly (get_mdm_loader, "
                "same batch size, sample limit, guidance scale, diffusion steps).",
    }
    print(json.dumps(result, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
