"""E0b -- reproduce MDM's published FID (0.544+/-.044) using the actual released checkpoint
and the actual vendored MDM code, at a REDUCED scale (128 samples, 1-2 replications instead of
the paper's ~1000 samples / 20 replications) because the author's own bundled eval log states
the full protocol took "about 12 Hrs" on their hardware -- infeasible to match at full scale on
this machine (CPU only, no dedicated GPU, Apple Silicon).

Deliberately uses their real argument parser (evaluation_parser(), via a constructed sys.argv)
rather than hand-building a config Namespace, so every hyperparameter (diffusion_steps=1000,
arch, latent_dim, guidance_param default 2.5, etc.) is loaded from the checkpoint's own bundled
args.json -- exactly the discipline that was missing when this project's original decode bug
happened (a hand-reconstructed config that runs cleanly and gives wrong numbers).

Run from third_party/motion-diffusion-model/ (its own relative-import structure requires this):
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e0b_mdm_reproduction.py --model-path <path/to/model475000.pt> \
        --num-samples-limit 128 --replication-times 1
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--num-samples-limit", type=int, default=128)
    ap.add_argument("--replication-times", type=int, default=1)
    ap.add_argument("--out-json", required=True)
    my_args = ap.parse_args()

    # Construct sys.argv for MDM's own evaluation_parser(), so args.json's real training
    # hyperparameters get loaded via their own load_args_from_model() -- not hand-reconstructed.
    sys.argv = [
        "eval_humanml.py",
        "--model_path", my_args.model_path,
        "--eval_mode", "debug",  # smallest built-in mode; we override its sample/rep counts below
        "--dataset", "humanml",
    ]

    from utils.parser_util import evaluation_parser
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from data_loaders.get_data import get_dataset_loader
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
    from utils.sampler_util import ClassifierFreeSampleModel
    import eval.eval_humanml as eh

    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32  # required by the R-Precision protocol, per their own code comment

    print("Loaded args from checkpoint's args.json:")
    print(json.dumps({k: v for k, v in vars(args).items()
                       if k in ("diffusion_steps", "arch", "latent_dim", "layers",
                                 "cond_mask_prob", "guidance_param", "dataset")}, indent=2))

    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    print("Device:", device)

    split = "test"
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                    num_frames=None, split=split, hml_mode="gt")
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size,
                                     num_frames=None, split=split, hml_mode="eval")

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    print(f"Loading checkpoint from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()

    num_samples_limit = my_args.num_samples_limit
    replication_times = my_args.replication_times

    eval_motion_loaders = {
        "vald": lambda: get_mdm_loader(
            args, model=model, diffusion=diffusion, batch_size=args.batch_size,
            ground_truth_loader=gen_loader, mm_num_samples=0, mm_num_repeats=0,
            max_motion_length=gt_loader.dataset.opt.max_motion_length,
            num_samples_limit=num_samples_limit, scale=args.guidance_param,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, device)

    log_path = Path(my_args.out_json).with_suffix(".log")
    mean_dict = eh.evaluation(
        eval_wrapper, gt_loader, eval_motion_loaders, str(log_path),
        replication_times, diversity_times=min(300, num_samples_limit), mm_num_times=0,
        run_mm=False, eval_platform=None,
    )

    result = {
        "experiment": "E0b",
        "description": "MDM published-FID reproduction, REDUCED SCALE "
                        f"(num_samples_limit={num_samples_limit}, replication_times={replication_times}) "
                        "vs. the paper's ~1000 samples / 20 replications -- infeasible at full "
                        "scale on this hardware (author's own log states ~12 Hrs for the full protocol).",
        "num_samples_limit": num_samples_limit,
        "replication_times": replication_times,
        "guidance_param": args.guidance_param,
        "diffusion_steps": args.diffusion_steps,
        "mean_dict": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in mean_dict.items()},
        "reference_paper_fid": {"value": 0.544, "ci": 0.044, "tolerance_pct5": [0.5168, 0.5712]},
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
