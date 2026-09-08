"""E1-pilot -- zero-training measurement of E1B's caption-truncation effect.

Reuses the already-validated evaluator (ground-truth R-Precision reproduces the published
reference to 0.06 sigma) to measure directly how much text-motion
retrieval signal the ORIGINAL project's own caption-truncation rule destroys -- same real test
motions, same evaluator, only the caption/token pairing changes between the two arms. No
training, no generation, no model at all.

Truncation rule reproduced faithfully from the archived original project's own code (READ-ONLY,
<ARCHIVE>/DL_T2P_IMPL.ipynb, HumanML3DProcessor._filter_for_static_poses): find the first
CCONJ/SCONJ POS tag, or the literal substring "/ADV then"|"/ADV after"|"/ADV before", in the
POS-tagged caption; truncate to the words before it; fall back to the first sentence if no match.
Reproduced INCLUDING a quirk in the original's own regex -- "/ADV then" as a literal substring
requires an ADV-tagged word immediately followed by the token "then" with no tag boundary in
between, which is not how POS-tagged tokens are actually formatted (each word carries its own
tag, e.g. "then/ADV", not a bare trailing "then"). This means the ADV branch of the original's
own regex very likely never fires as intended, and most samples fall through to the first-sentence
fallback. That is reported as a diagnostic finding about the ORIGINAL code, not silently fixed --
this script measures what the original actually did, not an idealized version of it.

One stated deviation: the original REJECTED any sample whose truncated text fell below 3 words.
This pilot cannot drop samples (every real motion still needs a caption pairing), so a
floor of the first 3 words is used instead for those cases, noted in the output.

This measures information loss in the TEXT ENCODER / retrieval space only, not generation
quality -- does not replace E1B.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    python3 ../../scripts/e1_pilot_caption_truncation.py --out-json <path>
"""
import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
sys.path.insert(0, os.path.abspath(MDM_ROOT))

CONJUNCTION_RE = re.compile(r"/CCONJ|/SCONJ|/ADV then|/ADV after|/ADV before")


def truncate_tokens_first_action_clause(caption, tokens):
    """tokens: list like ['a/DET', 'person/NOUN', 'is/AUX', 'walk/VERB', ...].

    Faithful port of the archived original's _filter_for_static_poses truncation logic,
    operating on the already-split token list (tokens[i] corresponds 1:1 with caption.split()[i]
    by construction of Text2MotionDatasetV2's own data loading, so no re-derivation needed).
    """
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
    floored = False
    if len(trunc_words) < 3:
        trunc_words, trunc_tokens = words[:3], tokens[:3]
        floored = True
    return " ".join(trunc_words), trunc_tokens, used_fallback, floored


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

    fixseed(my_args.seed)
    print("building full-caption (baseline) gt loader...")
    full_loader = get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                                      num_frames=None, split="test", hml_mode="gt")

    fixseed(my_args.seed)
    print("building truncated-caption gt loader (independent dataset instance, same split file)...")
    trunc_loader = get_dataset_loader(name="humanml", batch_size=my_args.batch_size,
                                       num_frames=None, split="test", hml_mode="gt")

    # Mutate the truncated loader's own data_dict in place -- an independent object from
    # full_loader's, per get_dataset_loader() decoding the cache file fresh each call.
    n_total, n_fallback, n_floored = 0, 0, 0
    word_count_before, word_count_after = [], []
    t2m_dataset = trunc_loader.dataset.t2m_dataset
    for key, entry in t2m_dataset.data_dict.items():
        for text_dict in entry["text"]:
            caption, tokens = text_dict["caption"], text_dict["tokens"]
            trunc_caption, trunc_tokens, used_fallback, floored = truncate_tokens_first_action_clause(
                caption, tokens
            )
            word_count_before.append(len(caption.split()))
            word_count_after.append(len(trunc_caption.split()))
            n_total += 1
            n_fallback += int(used_fallback)
            n_floored += int(floored)
            text_dict["caption"] = trunc_caption
            text_dict["tokens"] = trunc_tokens

    diagnostic = {
        "n_captions_total": n_total,
        "n_used_first_sentence_fallback": n_fallback,
        "pct_used_fallback": n_fallback / n_total if n_total else None,
        "n_floored_to_3_words": n_floored,
        "mean_words_before": sum(word_count_before) / len(word_count_before),
        "mean_words_after": sum(word_count_after) / len(word_count_after),
    }
    print("TRUNCATION DIAGNOSTIC:")
    print(json.dumps(diagnostic, indent=2))

    fixseed(my_args.seed)
    eval_wrapper = EvaluatorMDMWrapper("humanml", device)

    log_path = Path(my_args.out_json).with_suffix(".log")
    with open(log_path, "w") as f:
        fixseed(my_args.seed)
        motion_loaders = {"full_caption": full_loader, "truncated_caption": trunc_loader}
        match_score_dict, r_precision_dict, _ = evaluate_matching_score(eval_wrapper, motion_loaders, f)

    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else v

    r_prec_full = to_list(r_precision_dict["full_caption"])
    r_prec_trunc = to_list(r_precision_dict["truncated_caption"])
    top3_full = r_prec_full[2]
    top3_trunc = r_prec_trunc[2]

    result = {
        "experiment": "E1-pilot",
        "description": "Zero-training measurement of caption-truncation information loss, "
                       "same real test motions, faithful reproduction of the original project's "
                       "own truncation rule (see script docstring for the quirk found in its regex).",
        "batch_size": my_args.batch_size,
        "seed": my_args.seed,
        "truncation_diagnostic": diagnostic,
        "matching_score": {k: to_list(v) for k, v in match_score_dict.items()},
        "r_precision": {"full_caption": r_prec_full, "truncated_caption": r_prec_trunc},
        "r_precision_top3_full": top3_full,
        "r_precision_top3_truncated": top3_trunc,
        "r_precision_top3_drop": top3_full - top3_trunc,
        "reference_e0b_ground_truth_top3": 0.7969,
        "note": "This measures text-encoder/retrieval information loss only, not generation "
                "quality. Does not replace E1B.",
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
