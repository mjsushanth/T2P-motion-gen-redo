"""SUP-20260906-47 -- the single random-window draw (e1_pilot_followups.py) is one sample from
a stochastic treatment, not a deterministic arm like the prefix controls: reseeding only changes
which captions land in which batch for the prefix arms, but it changes the WINDOW PLACEMENT
ITSELF for random-window. Its across-seed swing (0.0124) is therefore mostly treatment variance,
not measurement noise -- the same pattern already flagged in E0a/`LANDMINES.md` (average over
repeated draws, MDM's own `repeat_time` convention).

This averages the random-window arm over N independent placement draws (same captions, same
per-key window length, same evaluator batch-shuffle seed held fixed across draws so only
placement varies), text re-encoding only, no model, no generation.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1_pilot_random_window_averaged.py --out-json <path>
"""
import argparse
import json
import os
import random as _random
import sys
from pathlib import Path

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))
sys.path.insert(0, os.path.dirname(__file__))

from e1_pilot_followups import (  # noqa: E402
    truncate_tokens_first_action_clause,
    make_deterministic_single_caption,
    restrict_dataset_to_keys,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batch-shuffle-seed", type=int, default=10)
    ap.add_argument("--n-draws", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    my_args = ap.parse_args()

    from utils.fixseed import fixseed
    from data_loaders.get_data import get_dataset_loader
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from eval.eval_humanml import evaluate_matching_score
    from utils import dist_util

    dist_util.setup_dist(0)
    device = dist_util.dev()
    print(f"device: {device}")

    def build_loader():
        return get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                                   num_frames=None, split="test", hml_mode="gt")

    fixseed(my_args.batch_shuffle_seed)
    full_loader = build_loader()
    fixseed(my_args.batch_shuffle_seed)
    rw_loader = build_loader()

    t2m_full = make_deterministic_single_caption(full_loader)
    t2m_rw = make_deterministic_single_caption(rw_loader)

    # Recompute per-key truncation length n and identify the could-vary subset, exactly as in
    # e1_pilot_followups.py, so this script's key set matches that run's 2,663-key restriction.
    per_key_n, could_vary_keys = {}, []
    for key in t2m_full.data_dict:
        caption = t2m_full.data_dict[key]["text"][0]["caption"]
        tokens = t2m_full.data_dict[key]["text"][0]["tokens"]
        _, trunc_tokens, _ = truncate_tokens_first_action_clause(caption, tokens)
        n = len(trunc_tokens)
        words = caption.split()
        per_key_n[key] = n
        if len(words) - n > 0:
            could_vary_keys.append(key)

    print(f"could-vary keys: {len(could_vary_keys)}")
    restrict_dataset_to_keys(rw_loader, set(could_vary_keys))
    t2m_rw = rw_loader.dataset.t2m_dataset
    print(f"restricted random-window dataset size: {len(rw_loader.dataset)}")

    fixseed(my_args.batch_shuffle_seed)
    eval_wrapper = EvaluatorMDMWrapper("humanml", device)

    draw_results = []
    log_path = Path(my_args.out_json).with_suffix(".log")
    with open(log_path, "w") as f:
        for draw_idx in range(my_args.n_draws):
            placement_seed = 1000 + draw_idx
            _random.seed(placement_seed)
            for key in t2m_rw.data_dict:
                caption = t2m_full.data_dict[key]["text"][0]["caption"]
                tokens = t2m_full.data_dict[key]["text"][0]["tokens"]
                n = per_key_n[key]
                words = caption.split()
                max_start = max(0, len(words) - n)
                start = _random.randint(0, max_start) if max_start > 0 else 0
                t2m_rw.data_dict[key]["text"][0]["caption"] = " ".join(words[start:start + n])
                t2m_rw.data_dict[key]["text"][0]["tokens"] = tokens[start:start + n]

            # Hold the batch-shuffle seed fixed across draws so only placement varies, isolating
            # treatment variance from the (already separately measured) batching variance.
            fixseed(my_args.batch_shuffle_seed)
            match_score, r_prec, _ = evaluate_matching_score(
                eval_wrapper, {"random_window_draw": rw_loader}, f
            )
            top3 = float(r_prec["random_window_draw"][2])
            print(f"draw {draw_idx} (placement_seed={placement_seed}): R-Prec-top3={top3:.4f}")
            draw_results.append({"draw_idx": draw_idx, "placement_seed": placement_seed, "r_precision_top3": top3})

    values = [d["r_precision_top3"] for d in draw_results]
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = variance ** 0.5

    result = {
        "experiment": "E1-pilot-random-window-averaged",
        "description": "SUP-20260906-47: average the random-window control over N independent "
                       "placement draws to separate treatment variance (which placement was "
                       "drawn) from measurement/batching noise, same pattern as MDM's own "
                       "repeat_time averaging.",
        "n_draws": my_args.n_draws,
        "batch_shuffle_seed": my_args.batch_shuffle_seed,
        "n_could_vary_keys": len(could_vary_keys),
        "draw_results": draw_results,
        "mean_r_precision_top3": mean_val,
        "std_r_precision_top3": std_val,
        "min_r_precision_top3": min(values),
        "max_r_precision_top3": max(values),
        "reference_length_matched_prefix_could_vary": 0.5456,
        "reference_single_draw_could_vary": 0.5339,
        "note": "batch-shuffle seed held fixed across draws; only window placement varies per "
                "draw, so std here is placement/treatment variance, not batching noise (that was "
                "measured separately in e1_pilot_followups.py's seed2 check).",
    }
    result["mean_effect_vs_prefix"] = result["reference_length_matched_prefix_could_vary"] - mean_val

    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "draw_results"}, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
