"""E1-pilot follow-ups -- three sharpenings of the E1-pilot result
(scripts/e1_pilot_caption_truncation.py), neither a correction, all needed before deciding
E1B's remaining scope.

Length-matched control: the rule-truncated captions are shorter (12.62 -> 8.01 words
on average) than the full captions. Some of the measured 0.145 R-Precision-top3 drop could be
"shorter captions retrieve worse" rather than "this specific selection rule is destructive."
Control: truncate the SAME full captions to the SAME per-caption word count via a content-neutral
rule (first N words, N = the rule-truncated caption's own length), re-measure.

Random-window control: both the rule and the first-N-words control keep the CAPTION PREFIX, so
neither separates "shorter" from "keeps the front of the sentence." Added a random contiguous
N-word WINDOW control (same length, different position) to actually isolate length from
position: if it scores close to the prefix controls, the effect really is length-driven; if it
scores materially worse, HumanML3D captions front-load their motion-relevant content and the
original's rule was accidentally preserving the useful part.

Conditional-effect control: the effect on captions the rule actually fires on. 44.8% of captions
fell through to the original's own first-sentence fallback (which changes little for already-short,
single-sentence captions). The corpus-wide 0.145 drop is diluted by that near-zero-effect share.
This computes the REAL conditional effect via an actual restricted retrieval run on only the
non-fallback subset (not an arithmetic estimate from the aggregate number) -- restricting both
the full-caption and truncated-caption arms to the same key subset before re-running
evaluate_matching_score.

One determinism change from e1_pilot_caption_truncation.py, stated plainly: this script fixes
each motion to its FIRST text entry (not Text2MotionDatasetV2's own random.choice per __getitem__
call), because the conditional-effect control needs a stable, known fallback-or-not label per
key, which random.choice would undermine. This means this script's own "full" and "truncated" arms
(recomputed here, not reused from the original E1-pilot run) will not be bit-identical to that
run's 0.8013/0.6563 -- expected to be close, reported as its own comparison point, not silently
substituted for the original.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1_pilot_followups.py --out-json <path>
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))

CONJUNCTION_RE = re.compile(r"/CCONJ|/SCONJ|/ADV then|/ADV after|/ADV before")


def truncate_tokens_first_action_clause(caption, tokens):
    pos_text = " ".join(tokens)
    words = caption.split()
    first_match = CONJUNCTION_RE.search(pos_text)
    used_fallback = first_match is None
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
    return " ".join(trunc_words), trunc_tokens, used_fallback


def make_deterministic_single_caption(loader):
    """Force each key to its FIRST text entry (not random.choice), so a fallback/non-fallback
    label per key is stable across the whole run."""
    t2m = loader.dataset.t2m_dataset
    for key, entry in t2m.data_dict.items():
        entry["text"] = [entry["text"][0]]
    return t2m


def restrict_dataset_to_keys(loader, keep_keys_set):
    """__len__ is len(data_dict) - pointer (NOT len(name_list)), so data_dict itself must be
    pruned too, or __getitem__'s name_list indexing goes out of range once __len__ reports the
    old, larger count."""
    t2m = loader.dataset.t2m_dataset
    orig_max_length = t2m.max_length
    filtered_name_list = tuple(k for k in t2m.name_list if k in keep_keys_set)
    t2m.data_dict = {k: t2m.data_dict[k] for k in filtered_name_list}
    t2m.name_list = filtered_name_list
    t2m.length_arr = np.array([t2m.data_dict[k]["length"] for k in filtered_name_list])
    t2m.pointer = 0
    t2m.reset_max_len(orig_max_length)
    return loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=10)
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

    fixseed(my_args.seed)
    full_loader = build_loader()
    fixseed(my_args.seed)
    trunc_loader = build_loader()
    fixseed(my_args.seed)
    lm_loader = build_loader()
    fixseed(my_args.seed)
    rw_loader = build_loader()

    t2m_full = make_deterministic_single_caption(full_loader)
    t2m_trunc = make_deterministic_single_caption(trunc_loader)
    t2m_lm = make_deterministic_single_caption(lm_loader)
    t2m_rw = make_deterministic_single_caption(rw_loader)

    import random as _random
    _random.seed(my_args.seed)

    fallback_keys, nonfallback_keys = [], []
    could_vary_keys, no_room_keys = [], []
    n_prefix_equals_full = 0
    for key in t2m_full.data_dict:
        caption, tokens = t2m_full.data_dict[key]["text"][0]["caption"], t2m_full.data_dict[key]["text"][0]["tokens"]
        trunc_caption, trunc_tokens, used_fallback = truncate_tokens_first_action_clause(caption, tokens)
        n = len(trunc_tokens)
        words = caption.split()
        lm_caption = " ".join(words[:n])
        lm_tokens = tokens[:n]

        # First-N-words and the rule both keep a PREFIX, so the length-matched control alone
        # cannot separate "shorter caption" from "keeps the front of the caption" -- add a random
        # contiguous N-word WINDOW, same length, different position, to actually isolate length
        # from position.
        max_start = max(0, len(words) - n)
        if max_start == 0:
            n_prefix_equals_full += 1
            start = 0
            no_room_keys.append(key)
        else:
            start = _random.randint(0, max_start)
            could_vary_keys.append(key)
        rw_words = words[start:start + n]
        rw_tokens = tokens[start:start + n]

        t2m_trunc.data_dict[key]["text"][0]["caption"] = trunc_caption
        t2m_trunc.data_dict[key]["text"][0]["tokens"] = trunc_tokens
        t2m_lm.data_dict[key]["text"][0]["caption"] = lm_caption
        t2m_lm.data_dict[key]["text"][0]["tokens"] = lm_tokens
        t2m_rw.data_dict[key]["text"][0]["caption"] = " ".join(rw_words)
        t2m_rw.data_dict[key]["text"][0]["tokens"] = rw_tokens

        (fallback_keys if used_fallback else nonfallback_keys).append(key)

    print(f"total keys: {len(t2m_full.data_dict)}, fallback: {len(fallback_keys)}, "
          f"non-fallback (rule actually fired): {len(nonfallback_keys)}, "
          f"keys where window==full (no room to move): {n_prefix_equals_full}")

    fixseed(my_args.seed)
    eval_wrapper = EvaluatorMDMWrapper("humanml", device)

    # --- Call A: full corpus, three arms (the length-matched control lives here) ---
    log_path_a = Path(my_args.out_json).with_suffix(".full_corpus.log")
    with open(log_path_a, "w") as f:
        fixseed(my_args.seed)
        motion_loaders_a = {
            "full_caption": full_loader,
            "truncated_caption": trunc_loader,
            "length_matched_control": lm_loader,
            "random_window_control": rw_loader,
        }
        match_score_a, r_prec_a, _ = evaluate_matching_score(eval_wrapper, motion_loaders_a, f)

    # --- Call B: the non-fallback-only subset, full vs truncated (the conditional-effect control) ---
    nonfallback_set = set(nonfallback_keys)
    restrict_dataset_to_keys(full_loader, nonfallback_set)
    restrict_dataset_to_keys(trunc_loader, nonfallback_set)
    print(f"restricted dataset sizes -- full: {len(full_loader.dataset)}, "
          f"trunc: {len(trunc_loader.dataset)}")

    log_path_b = Path(my_args.out_json).with_suffix(".nonfallback_subset.log")
    with open(log_path_b, "w") as f:
        fixseed(my_args.seed)
        motion_loaders_b = {
            "full_caption_nonfallback": full_loader,
            "truncated_caption_nonfallback": trunc_loader,
        }
        match_score_b, r_prec_b, _ = evaluate_matching_score(eval_wrapper, motion_loaders_b, f)

    # --- Call C: restrict the position controls to the 57.3% of keys that actually had room to
    # place a different window, so the position-effect null isn't mechanically diluted by the
    # 42.7% where random_window == length_matched by construction.
    could_vary_set = set(could_vary_keys)
    restrict_dataset_to_keys(lm_loader, could_vary_set)
    restrict_dataset_to_keys(rw_loader, could_vary_set)
    print(f"restricted (could-vary-only) dataset sizes -- length_matched: "
          f"{len(lm_loader.dataset)}, random_window: {len(rw_loader.dataset)}")

    log_path_c = Path(my_args.out_json).with_suffix(".could_vary_subset.log")
    with open(log_path_c, "w") as f:
        fixseed(my_args.seed)
        motion_loaders_c = {
            "length_matched_could_vary": lm_loader,
            "random_window_could_vary": rw_loader,
        }
        match_score_c, r_prec_c, _ = evaluate_matching_score(eval_wrapper, motion_loaders_c, f)

    # --- Call D: a noise-floor check local to this comparison -- E0b's ~0.016 floor was measured
    # on a different n (128) and a different score range (~0.65-0.80); re-shuffling the SAME
    # restricted (n=2663) datasets with a different seed gives a batching-noise estimate actually
    # local to this comparison's n and score range, instead of borrowing one from elsewhere.
    log_path_d = Path(my_args.out_json).with_suffix(".could_vary_subset_seed2.log")
    with open(log_path_d, "w") as f:
        fixseed(my_args.seed + 1)
        motion_loaders_d = {
            "length_matched_could_vary_seed2": lm_loader,
            "random_window_could_vary_seed2": rw_loader,
        }
        match_score_d, r_prec_d, _ = evaluate_matching_score(eval_wrapper, motion_loaders_d, f)

    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else v

    result = {
        "experiment": "E1-pilot-followups",
        "description": "Length-matched control + non-fallback-subset conditional effect, both "
                       "real retrieval re-runs, not arithmetic estimates from the original "
                       "E1-pilot's aggregate numbers.",
        "seed": my_args.seed,
        "n_total_keys": len(fallback_keys) + len(nonfallback_keys),
        "n_fallback_keys": len(fallback_keys),
        "n_nonfallback_keys": len(nonfallback_keys),
        "pct_nonfallback": len(nonfallback_keys) / (len(fallback_keys) + len(nonfallback_keys)),
        "n_keys_window_equals_full_no_room_to_move": n_prefix_equals_full,
        "full_corpus_r_precision": {
            "full_caption": to_list(r_prec_a["full_caption"]),
            "truncated_caption": to_list(r_prec_a["truncated_caption"]),
            "length_matched_control": to_list(r_prec_a["length_matched_control"]),
            "random_window_control": to_list(r_prec_a["random_window_control"]),
        },
        "full_corpus_matching_score": {k: to_list(v) for k, v in match_score_a.items()},
        "nonfallback_subset_r_precision": {
            "full_caption_nonfallback": to_list(r_prec_b["full_caption_nonfallback"]),
            "truncated_caption_nonfallback": to_list(r_prec_b["truncated_caption_nonfallback"]),
        },
        "nonfallback_subset_matching_score": {k: to_list(v) for k, v in match_score_b.items()},
        "n_could_vary_keys": len(could_vary_keys),
        "n_no_room_keys": len(no_room_keys),
        "pct_could_vary": len(could_vary_keys) / (len(could_vary_keys) + len(no_room_keys)),
        "could_vary_subset_r_precision": {
            "length_matched_could_vary": to_list(r_prec_c["length_matched_could_vary"]),
            "random_window_could_vary": to_list(r_prec_c["random_window_could_vary"]),
        },
        "could_vary_subset_matching_score": {k: to_list(v) for k, v in match_score_c.items()},
        "could_vary_subset_seed2_r_precision": {
            "length_matched_could_vary_seed2": to_list(r_prec_d["length_matched_could_vary_seed2"]),
            "random_window_could_vary_seed2": to_list(r_prec_d["random_window_could_vary_seed2"]),
        },
        "original_e1_pilot_reference": {
            "r_precision_top3_full": 0.8013, "r_precision_top3_truncated": 0.6563, "drop": 0.1450,
            "note": "recomputed here under deterministic single-caption-per-key selection, "
                    "not random.choice -- expected close but not bit-identical",
        },
    }

    top3 = lambda arr: arr[2] if arr is not None else None
    result["summary"] = {
        "corpus_wide_drop_recomputed": top3(r_prec_a["full_caption"]) - top3(r_prec_a["truncated_caption"]),
        "length_matched_control_drop": top3(r_prec_a["full_caption"]) - top3(r_prec_a["length_matched_control"]),
        "random_window_control_drop": top3(r_prec_a["full_caption"]) - top3(r_prec_a["random_window_control"]),
        "rule_specific_drop_beyond_length": top3(r_prec_a["length_matched_control"]) - top3(r_prec_a["truncated_caption"]),
        "position_effect_prefix_minus_random_window": top3(r_prec_a["random_window_control"]) - top3(r_prec_a["length_matched_control"]),
        "conditional_drop_on_nonfallback_subset": top3(r_prec_b["full_caption_nonfallback"]) - top3(r_prec_b["truncated_caption_nonfallback"]),
        "position_effect_on_could_vary_subset_only": top3(r_prec_c["random_window_could_vary"]) - top3(r_prec_c["length_matched_could_vary"]),
        "lm_batching_noise_seed1_vs_seed2": top3(r_prec_d["length_matched_could_vary_seed2"]) - top3(r_prec_c["length_matched_could_vary"]),
        "rw_batching_noise_seed1_vs_seed2": top3(r_prec_d["random_window_could_vary_seed2"]) - top3(r_prec_c["random_window_could_vary"]),
    }

    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
