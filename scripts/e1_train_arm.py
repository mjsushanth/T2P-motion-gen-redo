"""E1 arm trainer -- generalizes scripts/e1a_power_check.py to run either E1A (control: full
caption throughout) or E1B (first-action-clause truncation applied to BOTH training captions and
generation-conditioning captions, per docs/DECISIONS.md D-23's redesigned A/B/C ladder).

--arm a: full caption -> full sequence (the control; E1A's power check already produced one real
         result this way, R-Precision-top3=0.2969 vs 0.7950 ground truth, LEDGER Item 29).
--arm b: first-action-clause-only caption -> full sequence. Truncation is applied to the TRAIN
         loader's captions/tokens (what the model learns to condition on) AND to the generation
         loader's captions/tokens (what conditions sampling and what the retrieval evaluator
         scores the generated motion against) -- mirroring the original project's actual defect,
         where truncated captions were used end-to-end, not just at one stage. The ground-truth
         reference loader (used only as the fixed baseline column in evaluate_matching_score/FID)
         is left with FULL captions unchanged, same as every other arm run in this project.

Both arms train on the disjoint materialized TRAIN split and evaluate on the materialized TEST
split (docs/DECISIONS.md D-25 / review SUP-20260906-37) -- no memorisation confound.

Includes the diversity_times off-by-one fix (review SUP-20260906-8/E1A-power's own crash,
LEDGER Item 29): diversity_times must be < the generated set size, not <=.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1_train_arm.py --arm b --num-steps 3000 --out-json <path>
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))

CONJUNCTION_RE = re.compile(r"/CCONJ|/SCONJ|/ADV then|/ADV after|/ADV before")


def truncate_tokens_first_action_clause(caption, tokens):
    """Same faithful port of the archived original's truncation rule used throughout the
    E1-pilot (scripts/e1_pilot_caption_truncation.py, scripts/e1_pilot_followups.py)."""
    pos_text = " ".join(tokens)
    words = caption.split()
    first_match = CONJUNCTION_RE.search(pos_text)
    if first_match is not None:
        pos_before_conj = pos_text[: first_match.start()].count(" ")
        trunc_words = words[:pos_before_conj]
        trunc_tokens = tokens[:pos_before_conj]
    else:
        sentences = re.split(r"[.!?]", caption)
        first_sentence = sentences[0].strip() if sentences else caption
        n = len(first_sentence.split())
        trunc_words = words[:n]
        trunc_tokens = tokens[:n]
    if len(trunc_words) < 3:
        trunc_words, trunc_tokens = words[:3], tokens[:3]
    return " ".join(trunc_words), trunc_tokens


def rescore_against_truncated_captions(motion_loader, eval_wrapper, arm_label):
    """SUP-20260906-49: E1A is scored against full captions (0.2969) but E1B will be scored
    against truncated ones, so the raw A-vs-B gap conflates model degradation with "truncated
    captions are intrinsically harder to retrieve against" -- a text-side effect the E1-pilot
    already measured on real motions (0.145), but not in this model's much-lower operating range
    (~0.30 vs ~0.80), where it will not transfer at the same magnitude. This control holds the
    SAME generated motions fixed and rescopes only the retrieval caption, isolating "caption
    retrievability alone, in this model's actual operating range" from any model difference.

    generated_motion[i]['tokens'] is the WRAPPED, padded token list (sos/OTHER ... eos/OTHER ...
    unk/OTHER padding) already used to compute embeddings; must unwrap, truncate the real
    content, then re-wrap to the same fixed length before rescoring, or the evaluator's
    w_vectorizer lookup and cap_len bookkeeping break.
    """
    from eval.eval_humanml import evaluate_matching_score

    for item in motion_loader.dataset.generated_motion:
        tokens, cap_len, caption = item["tokens"], item["cap_len"], item["caption"]
        real_tokens = tokens[1:cap_len - 1]  # strip sos/OTHER ... eos/OTHER wrapper
        trunc_caption, trunc_tokens = truncate_tokens_first_action_clause(caption, real_tokens)
        total_len = len(tokens)
        new_wrapped = ["sos/OTHER"] + trunc_tokens + ["eos/OTHER"]
        new_cap_len = len(new_wrapped)
        if new_cap_len < total_len:
            new_wrapped = new_wrapped + ["unk/OTHER"] * (total_len - new_cap_len)
        else:
            new_wrapped = new_wrapped[:total_len]
            new_cap_len = total_len
        item["caption"] = trunc_caption
        item["tokens"] = new_wrapped
        item["cap_len"] = new_cap_len

    match_score, r_prec, _ = evaluate_matching_score(
        eval_wrapper, {arm_label: motion_loader}, open(os.devnull, "w")
    )
    return match_score[arm_label], r_prec[arm_label]


def truncate_all_captions_in_loader(loader):
    """Mutates every text entry (not just the first) for every key -- unlike the E1-pilot's
    retrieval-only analysis, training draws via Text2MotionDatasetV2's own random.choice per
    __getitem__, so every entry in a multi-caption key must be truncated, not just one."""
    t2m = loader.dataset.t2m_dataset
    n_captions = 0
    for key, entry in t2m.data_dict.items():
        for text_dict in entry["text"]:
            trunc_caption, trunc_tokens = truncate_tokens_first_action_clause(
                text_dict["caption"], text_dict["tokens"]
            )
            text_dict["caption"] = trunc_caption
            text_dict["tokens"] = trunc_tokens
            n_captions += 1
    return n_captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"], required=True)
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
        "e1_train_arm",
        "--save_dir", str(Path(my_args.out_json).parent / f"e1{my_args.arm}_scratch_save_dir"),
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
    print(f"device: {device}, arm: {my_args.arm}")

    print(f"loading training data (split={my_args.train_split})...")
    train_data = get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                                     num_frames=None, split=my_args.train_split, hml_mode="train")
    print(f"train dataset size: {len(train_data.dataset)} sequences")

    n_train_captions_truncated = 0
    if my_args.arm == "b":
        n_train_captions_truncated = truncate_all_captions_in_loader(train_data)
        print(f"arm B: truncated {n_train_captions_truncated} training caption entries "
              f"(first-action-clause rule)")

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

    print("generating + evaluating...")
    model.eval()
    guidance_param = 2.5
    sample_model = ClassifierFreeSampleModel(model) if guidance_param != 1 else model
    sample_model.to(device)

    gt_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                    split=my_args.eval_split, hml_mode="gt")
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                     split=my_args.eval_split, hml_mode="eval")

    n_gen_captions_truncated = 0
    if my_args.arm == "b":
        # Truncate the generation-conditioning/scoring captions too -- mirrors the original
        # defect end-to-end (truncated captions used for both training and use), and keeps the
        # R-Precision instrument internally consistent (it conditions generation AND scores the
        # result from the SAME stored caption/tokens, per eval_humanml.evaluate_matching_score).
        n_gen_captions_truncated = truncate_all_captions_in_loader(gen_loader)
        print(f"arm B: truncated {n_gen_captions_truncated} generation-conditioning caption entries")

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

    arm_name = f"e1{my_args.arm}"
    eval_motion_loaders = {arm_name: lambda: (motion_loader, mm_motion_loader)}
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, device)

    log_path = Path(my_args.out_json).with_suffix(".log")
    mean_dict = eh.evaluation(
        eval_wrapper, gt_loader, eval_motion_loaders, str(log_path),
        replication_times=1, diversity_times=min(300, my_args.num_samples_limit - 1),
        mm_num_times=0, run_mm=False, eval_platform=None,
    )

    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else v

    truncated_rescore = None
    if my_args.arm == "a":
        # SUP-20260906-49's control: same generated motions, same model, rescored against
        # truncated captions -- isolates caption-retrievability-alone in this model's operating
        # range, before any E1B comparison is written up. Cheap: text re-encoding only, the
        # already-generated motion tensors are reused as-is (no regeneration).
        print("rescoring E1A's own generations against truncated captions "
              "(SUP-20260906-49 control)...")
        rescore_match, rescore_rprec = rescore_against_truncated_captions(
            motion_loader, eval_wrapper, "e1a_truncated_rescore"
        )
        truncated_rescore = {
            "matching_score": float(rescore_match),
            "r_precision": to_list(rescore_rprec),
            "r_precision_top3": float(to_list(rescore_rprec)[2]),
        }
        print(f"  e1a_truncated_rescore R-Precision-top3: {truncated_rescore['r_precision_top3']:.4f}")

    r_prec_arm = to_list(mean_dict.get(f"R_precision_{arm_name}"))
    r_prec_gt = to_list(mean_dict.get("R_precision_ground truth"))
    fid_arm = to_list(mean_dict.get(f"FID_{arm_name}"))

    chance_level = 3.0 / 32.0
    r_prec_top3_arm = r_prec_arm[2] if r_prec_arm is not None else None

    result = {
        "experiment": f"E1{my_args.arm.upper()}",
        "arm": my_args.arm,
        "description": ("Control: full caption -> full sequence throughout." if my_args.arm == "a"
                        else "Truncated (first-action-clause) caption -> full sequence, "
                             "truncation applied to both training and generation-conditioning, "
                             "per docs/DECISIONS.md D-23."),
        "num_training_steps": my_args.num_steps,
        "seed": my_args.seed,
        "train_split": my_args.train_split,
        "eval_split": my_args.eval_split,
        "batch_size": my_args.batch_size,
        "num_samples_limit": my_args.num_samples_limit,
        "n_params_millions": n_params / 1e6,
        "n_train_captions_truncated": n_train_captions_truncated,
        "n_gen_captions_truncated": n_gen_captions_truncated,
        "training_wall_clock_seconds": training_wall_clock_s,
        "generation_wall_clock_seconds": generation_wall_clock_s,
        "loss_log": loss_log,
        "mean_dict": {k: to_list(v) for k, v in mean_dict.items()},
        "e1a_truncated_rescore_control": truncated_rescore,
        "chance_level_r_precision_top3": chance_level,
        f"r_precision_top3_{arm_name}": r_prec_top3_arm,
        "r_precision_top3_ground_truth": r_prec_gt[2] if r_prec_gt is not None else None,
        f"fid_{arm_name}_vs_ground_truth": fid_arm,
        "note": f"trained on split={my_args.train_split}, evaluated on split={my_args.eval_split} "
                "(disjoint materialized subsets, per review SUP-20260906-37).",
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "loss_log"}, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
