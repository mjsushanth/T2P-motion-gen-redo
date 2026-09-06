"""E0 -- validate the vendored Guo et al. evaluation harness (docs/DECISIONS.md D-03 gate).

Sanity check, NOT the final D-03 result: since no generative model exists yet in this
repository, this cannot reproduce a published *generated-model* FID (e.g. MDM's 0.544). What it
CAN do is confirm the harness produces sane numbers on real data: split real HumanML3D test-split
motions into two disjoint subsets, treat one as "reference" and one as "candidate" exactly as
`final_evaluations.py`'s own `evaluate_fid` does, and check FID is small (real vs real) and
R-Precision/matching score are in a sane range (matched text-motion pairs should rank highly).

Run with this repo's third_party/text-to-motion/ on PYTHONPATH, e.g. from T2P-Reboot/:
    PYTHONPATH=third_party/text-to-motion python3 scripts/e0_evaluator_sanity_check.py
No sys.path manipulation in this file, per project convention -- PYTHONPATH is an invocation-time
concern, not a code-level hack.
"""
import json
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from networks.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from utils.metrics import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_activation_statistics,
    calculate_frechet_distance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
STATS_DIR = REPO_ROOT / "checkpoints" / "t2m_dataset_stats"
GLOVE_DIR = REPO_ROOT / "third_party" / "text-to-motion" / "glove"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "e0"

UNIT_LENGTH = 4
MAX_MOTION_LENGTH = 196
MAX_TEXT_LEN = 20
MIN_MOTION_LEN = 40
MAX_MOTION_LEN_FILTER = 200
N_SAMPLES = 6000  # exceeds the test split's actual size (~4384 seqs) -- uses the whole split
R_PRECISION_BATCH_SIZE = 32  # matches the paper's protocol: 1 correct + 31 distractors per batch
SEED = int(os.environ.get("E0_SEED", "0"))


def build_opt(device: torch.device) -> Namespace:
    opt = Namespace()
    opt.dataset_name = "t2m"
    opt.device = device
    opt.checkpoints_dir = str(CHECKPOINTS_DIR)
    opt.unit_length = UNIT_LENGTH
    opt.max_motion_length = MAX_MOTION_LENGTH
    opt.max_text_len = MAX_TEXT_LEN
    # Not set by EvaluatorModelWrapper.__init__ itself; repo defaults (options/base_options.py),
    # confirmed against the downloaded checkpoint's own tensor shapes (512-dim throughout).
    opt.dim_movement_enc_hidden = 512
    opt.dim_movement_latent = 512
    return opt


def prepare_sample(motion: np.ndarray, caption_str: str, mean: np.ndarray, std: np.ndarray,
                    w_vectorizer: WordVectorizer):
    parts = caption_str.strip().split("#")
    if len(parts) < 2:
        return None
    tokens = parts[1].split(" ")
    if len(tokens) < MAX_TEXT_LEN:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        tokens = tokens[:MAX_TEXT_LEN]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)

    pos_one_hots, word_embeddings = [], []
    for token in tokens:
        try:
            word_emb, pos_oh = w_vectorizer[token]
        except (KeyError, ValueError):
            word_emb, pos_oh = w_vectorizer["unk/OTHER"]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)

    m_length = (len(motion) // UNIT_LENGTH) * UNIT_LENGTH
    if m_length == 0:
        return None
    idx = random.randint(0, len(motion) - m_length)
    motion = motion[idx: idx + m_length]
    motion = (motion - mean) / std
    if m_length < MAX_MOTION_LENGTH:
        motion = np.concatenate(
            [motion, np.zeros((MAX_MOTION_LENGTH - m_length, motion.shape[1]))], axis=0
        )
    else:
        motion = motion[:MAX_MOTION_LENGTH]
        m_length = MAX_MOTION_LENGTH
    return word_embeddings, pos_one_hots, sent_len, motion, m_length


def collate(batch):
    # Matches the official collate_fn (data/dataset.py): sort by sent_len (index 2 here)
    # descending, since TextEncoderBiGRUCo's pack_padded_sequence requires it and is not
    # sorted internally before that call.
    batch = sorted(batch, key=lambda b: b[2], reverse=True)
    word_embeddings = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    pos_one_hots = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    sent_lens = torch.tensor([b[2] for b in batch])
    motions = torch.from_numpy(np.stack([b[3] for b in batch])).float()
    m_lens = torch.tensor([b[4] for b in batch])
    return word_embeddings, pos_one_hots, sent_lens, motions, m_lens


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    opt = build_opt(device)
    print("Loading EvaluatorModelWrapper from", CHECKPOINTS_DIR / "t2m" / "text_mot_match")
    wrapper = EvaluatorModelWrapper(opt)

    mean = np.load(STATS_DIR / "Mean.npy")
    std = np.load(STATS_DIR / "Std.npy")
    w_vectorizer = WordVectorizer(str(GLOVE_DIR), "our_vab")

    print(f"Streaming HumanML3D test split (target {N_SAMPLES} samples)...")
    ds = load_dataset("TeoGchx/HumanML3D", split="test", streaming=True)
    samples = []
    n_seen = 0
    n_skipped_len = 0
    n_skipped_caption = 0
    for ex in ds:
        n_seen += 1
        motion = np.asarray(ex["motion"], dtype=np.float64)
        if motion.shape[0] < MIN_MOTION_LEN or motion.shape[0] >= MAX_MOTION_LEN_FILTER or motion.shape[1] != 263:
            n_skipped_len += 1
            continue
        prepped = prepare_sample(motion, ex["caption"], mean, std, w_vectorizer)
        if prepped is None:
            n_skipped_caption += 1
            continue
        samples.append(prepped)
        if len(samples) >= N_SAMPLES:
            break
    print(f"Seen {n_seen} rows -> prepared {len(samples)} samples "
          f"(skipped {n_skipped_len} for length, {n_skipped_caption} for caption parsing)")

    if len(samples) < 2 * R_PRECISION_BATCH_SIZE:
        raise RuntimeError(f"Too few usable samples ({len(samples)}) for a meaningful check")

    half = len(samples) // 2
    subset_a = samples[:half]
    subset_b = samples[half: 2 * half]
    print(f"Split into subset A ({len(subset_a)}) and subset B ({len(subset_b)}) -- disjoint real motions")

    # --- R-Precision / matching score: batched by R_PRECISION_BATCH_SIZE (paper protocol: 32
    # candidates per batch = 1 correct + 31 distractors), averaged across batches, on subset A ---
    all_motion_embeddings_a = []
    matching_scores = []
    top_k_sums = np.zeros(3)
    n_batches = 0
    for i in range(0, len(subset_a) - R_PRECISION_BATCH_SIZE + 1, R_PRECISION_BATCH_SIZE):
        mini_batch = subset_a[i: i + R_PRECISION_BATCH_SIZE]
        word_emb, pos_oh, sent_len, motions, m_len = collate(mini_batch)
        with torch.no_grad():
            text_emb, motion_emb = wrapper.get_co_embeddings(
                word_embs=word_emb, pos_ohot=pos_oh, cap_lens=sent_len, motions=motions, m_lens=m_len
            )
        all_motion_embeddings_a.append(motion_emb.cpu().numpy())
        dist_mat = euclidean_distance_matrix(text_emb.cpu().numpy(), motion_emb.cpu().numpy())
        matching_scores.append(dist_mat.trace() / dist_mat.shape[0])
        argsmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        top_k_sums += top_k_mat.sum(axis=0)
        n_batches += 1
    matching_score = float(np.mean(matching_scores))
    r_precision = (top_k_sums / (n_batches * R_PRECISION_BATCH_SIZE)).tolist()
    motion_emb_a_full = np.concatenate(all_motion_embeddings_a, axis=0)
    n_r_precision_samples = n_batches * R_PRECISION_BATCH_SIZE

    # --- FID between two DISJOINT real subsets, using as many samples as available (subset B via
    # get_motion_embeddings in chunks, to keep memory/GRU batch size reasonable) ---
    motion_emb_b_chunks = []
    chunk = 64
    for i in range(0, len(subset_b), chunk):
        mini_batch = subset_b[i: i + chunk]
        _, _, _, motions, m_len = collate(mini_batch)
        with torch.no_grad():
            motion_emb_b_chunks.append(wrapper.get_motion_embeddings(motions=motions, m_lens=m_len).cpu().numpy())
    motion_emb_b_full = np.concatenate(motion_emb_b_chunks, axis=0)

    mu_a, cov_a = calculate_activation_statistics(motion_emb_a_full)
    mu_b, cov_b = calculate_activation_statistics(motion_emb_b_full)
    fid = float(calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b))
    print(f"FID computed on {motion_emb_a_full.shape[0]} (A) vs {motion_emb_b_full.shape[0]} (B) "
          f"samples, embedding dim {motion_emb_a_full.shape[1]}")

    result = {
        "experiment": "E0",
        "description": "Evaluator harness sanity check: real-vs-real FID and matched-pair "
                        "R-Precision/matching-score. NOT a reproduction of a published "
                        "generated-model number (no generative model exists in this repo yet).",
        "n_samples_subset_a_available": len(subset_a),
        "n_samples_subset_a_used_for_r_precision": n_r_precision_samples,
        "r_precision_batch_size": R_PRECISION_BATCH_SIZE,
        "n_r_precision_batches": n_batches,
        "n_samples_subset_b": len(subset_b),
        "fid_embedding_dim": int(motion_emb_a_full.shape[1]),
        "seed": SEED,
        "matching_score": matching_score,
        "r_precision_top1_2_3": r_precision,
        "fid_real_vs_real_disjoint_subsets": fid,
        "checkpoint_provenance": "third-party HF re-upload (Tevior/text_mot_match), "
                                  "architecture-verified not cryptographically verified -- "
                                  "see third_party/text-to-motion/PATCHES.md",
        "reference_paper_real_row": {"r_precision_top3": 0.797, "fid": 0.002,
                                     "note": "paper's ground-truth-vs-itself row on the FULL "
                                             "test set, not directly comparable to this "
                                             "smaller disjoint-subset check"},
    }
    print(json.dumps(result, indent=2))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"e0_evaluator_sanity_check_record_seed{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
