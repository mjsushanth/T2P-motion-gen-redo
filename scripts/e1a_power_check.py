"""E1A power check -- its chance threshold and pass/fail criterion were fixed before this was run.

Trains MDM's real architecture from scratch for a small number of steps (E1A: full caption,
full-sequence target -- the control arm of the redesigned E1 ladder, `docs/DECISIONS.md` D-23),
then generates and evaluates R-Precision-top3 against the chance threshold fixed in advance
(3/32 = 0.09375, per `docs/DECISIONS.md` D-25's R-Precision-decisive regime for E1/E2).

Not the E1 result itself -- a cheap (~1.9h training + ~0.65h generation/eval) check for whether
this training budget gives E1's A-vs-B comparison any power at all, before spending the full
matrix's wall-clock.

Trains on the materialized TRAIN split, evaluates on the materialized TEST split: training and
evaluating on the same subset would let an above-chance R-Precision result reflect memorisation
of the training pairs rather than generalisable text conditioning -- keeping them disjoint is
needed for a check whose only job is detecting real learning, not a milder version of the same
signal. Requires the train split to already be materialized via
`materialize_humanml3d_test_subset.py --split train`.

Reuses vendored MDM machinery throughout (train_args() parser, create_model_and_diffusion,
ClassifierFreeSampleModel, get_mdm_loader, EvaluatorMDMWrapper, eval_humanml.evaluation) --
the same pattern as scripts/e0b_mdm_reproduction.py and scripts/e1_training_feasibility_probe.py.
No new denoiser, no reimplemented training loop beyond the minimal step function already
validated in the feasibility probe.

Run from third_party/motion-diffusion-model/ (its own relative-import structure requires this):
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1a_power_check.py --num-steps 3000 --out-json <path>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-samples-limit", type=int, default=128)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--eval-split", type=str, default="test")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--log-every", type=int, default=200)
    my_args = ap.parse_args()

    scratch_argv = [
        "e1a_power_check",
        "--save_dir", str(Path(my_args.out_json).parent / "e1a_scratch_save_dir"),
        "--overwrite",
        "--dataset", "humanml",
        "--batch_size", str(my_args.batch_size),
        "--device", "0",
        "--seed", str(my_args.seed),
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
    from diffusion.resample import create_named_schedule_sampler
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
    import eval.eval_humanml as eh
    import torch

    fixseed(my_args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print(f"device: {device}")

    print(f"loading training data (split={my_args.train_split})...")
    train_data = get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                                     num_frames=None, split=my_args.train_split, hml_mode="train")
    print(f"train dataset size: {len(train_data.dataset)} sequences")

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_data)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters_wo_clip())
    print(f"total trainable params (excluding CLIP): {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    data_iter = iter(train_data)

    def next_batch():
        nonlocal data_iter
        try:
            motion, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            motion, cond = next(data_iter)
        motion = motion.to(device)
        cond["y"] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond["y"].items()}
        return motion, cond

    def run_one_step():
        motion, cond = next_batch()
        opt.zero_grad()
        t, weights = schedule_sampler.sample(motion.shape[0], device)
        losses = diffusion.training_losses(model, motion, t, model_kwargs=cond, dataset=train_data.dataset)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())

    print(f"training {my_args.num_steps} steps...")
    model.train()
    t_train_start = time.perf_counter()
    loss_log = []
    for step in range(my_args.num_steps):
        loss_val = run_one_step()
        if step % my_args.log_every == 0 or step == my_args.num_steps - 1:
            elapsed = time.perf_counter() - t_train_start
            print(f"  step {step}/{my_args.num_steps}: loss={loss_val:.5f}, elapsed={elapsed:.1f}s")
            loss_log.append({"step": step, "loss": loss_val, "elapsed_s": elapsed})
    t_train_end = time.perf_counter()
    training_wall_clock_s = t_train_end - t_train_start
    print(f"training done: {training_wall_clock_s:.1f}s ({training_wall_clock_s/3600:.3f}h)")

    print("generating + evaluating (E0b pattern, same trained model, no checkpoint round-trip)...")
    model.eval()
    guidance_param = 2.5
    sample_model = ClassifierFreeSampleModel(model) if guidance_param != 1 else model
    sample_model.to(device)

    gt_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                    split=my_args.eval_split, hml_mode="gt")
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                     split=my_args.eval_split, hml_mode="eval")

    t_gen_start = time.perf_counter()
    motion_loader, mm_motion_loader = get_mdm_loader(
        args, model=sample_model, diffusion=diffusion, batch_size=32,
        ground_truth_loader=gen_loader, mm_num_samples=0, mm_num_repeats=0,
        max_motion_length=gt_loader.dataset.opt.max_motion_length,
        num_samples_limit=my_args.num_samples_limit, scale=guidance_param,
    )
    t_gen_end = time.perf_counter()
    generation_wall_clock_s = t_gen_end - t_gen_start
    print(f"generation done: {generation_wall_clock_s:.1f}s ({generation_wall_clock_s/3600:.3f}h)")

    eval_motion_loaders = {"e1a": lambda: (motion_loader, mm_motion_loader)}
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, device)

    log_path = Path(my_args.out_json).with_suffix(".log")
    mean_dict = eh.evaluation(
        eval_wrapper, gt_loader, eval_motion_loaders, str(log_path),
        # diversity_times must be < activation.shape[0] (calculate_diversity's own assert), not
        # <=, so a value equal to num_samples_limit crashes once the generated set reaches
        # exactly that size -- the same off-by-one that crashed E0b and this script's own first
        # real run. Fixed here rather than deferred a third time.
        replication_times=1, diversity_times=min(300, my_args.num_samples_limit - 1),
        mm_num_times=0, run_mm=False, eval_platform=None,
    )

    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else v

    r_prec_e1a = to_list(mean_dict.get("R_precision_e1a"))
    r_prec_gt = to_list(mean_dict.get("R_precision_ground truth"))
    fid_e1a = to_list(mean_dict.get("FID_e1a"))

    chance_level = 3.0 / 32.0
    r_prec_top3_e1a = r_prec_e1a[2] if r_prec_e1a is not None else None
    above_chance = (r_prec_top3_e1a is not None) and (r_prec_top3_e1a > chance_level)

    result = {
        "experiment": "E1A-power",
        "description": "Power check, chance threshold fixed in advance: does E1A (full caption "
                       "-> full sequence) trained for num_steps learn text conditioning at all.",
        "num_training_steps": my_args.num_steps,
        "seed": my_args.seed,
        "train_split": my_args.train_split,
        "eval_split": my_args.eval_split,
        "batch_size": my_args.batch_size,
        "num_samples_limit": my_args.num_samples_limit,
        "n_params_millions": n_params / 1e6,
        "training_wall_clock_seconds": training_wall_clock_s,
        "generation_wall_clock_seconds": generation_wall_clock_s,
        "loss_log": loss_log,
        "mean_dict": {k: to_list(v) for k, v in mean_dict.items()},
        "chance_level_r_precision_top3": chance_level,
        "r_precision_top3_e1a": r_prec_top3_e1a,
        "r_precision_top3_ground_truth": r_prec_gt[2] if r_prec_gt is not None else None,
        "fid_e1a_vs_ground_truth": fid_e1a,
        "gate_result": "above_chance" if above_chance else "at_or_near_chance",
        "note": f"trained on split={my_args.train_split}, evaluated on split={my_args.eval_split} "
                "(disjoint materialized subsets) -- an above-chance "
                "result here reflects held-out generalisation, not memorisation of the training "
                "captions/motions.",
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "loss_log"}, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
