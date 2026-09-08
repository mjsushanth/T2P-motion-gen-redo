"""E3 follow-up: does randomizing batch-of-32 assignment change Guo R-Precision-top3?

Tests whether the discrepancy between notebooks/02's own n=128 measurement of this checkpoint
(Guo R-Precision-top3 = 0.7578) and E3's own n=4,640 measurement (0.6172, a 3.71 sigma gap using
the n=128 estimate's own binomial SE) is a batch-composition artifact -- R-Precision ranks each
sample against 31 decoys drawn from its own batch, so if consecutive samples in loader order are
more mutually similar than a random draw would be, decoys get easier and R-Precision rises for
that reason alone. A real compositional drift motivated the test: E3's own first 1,696 captions
run 60.6% spatial, the next 2,304 run 57.1%.

Uses the SAME already-computed embeddings from E3's cached generations -- no regeneration, no new
diffusion sampling. Reshuffles the full 4,640-sample pool into new batches of 32, twenty times,
and compares against the as-generated (natural loader) order. See
docs/EXPERIMENT_DESIGN_E3.md section 9.5 for the result and its interpretation.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    PYTHONPATH=. python3 ../../scripts/e3_batch_shuffle_diagnostic.py \
        --gen-dir ../../artifacts/e3 --out-json ../../artifacts/e3/shuffle_diagnostic.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-trials", type=int, default=20)
    my_args = ap.parse_args()

    MDM_ROOT = os.path.abspath(os.getcwd())  # run from third_party/motion-diffusion-model/
    sys.path.insert(0, MDM_ROOT)
    sys.path.insert(0, os.path.join(MDM_ROOT, "..", "..", "scripts"))

    from e3_score import load_generated_batches, is_spatial, r_precision_batches
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

    print("Loading cached generated batches...")
    motions_raw, lengths, captions, tokens_list, cap_lens = load_generated_batches(my_args.gen_dir)
    n_gen = len(captions)
    print(f"{n_gen} generated samples loaded")

    mdm_mean = np.load(os.path.join(MDM_ROOT, "dataset", "HumanML3D", "Mean.npy"))
    mdm_std = np.load(os.path.join(MDM_ROOT, "dataset", "HumanML3D", "Std.npy"))
    motions = (motions_raw * mdm_std[None, None, :] + mdm_mean[None, None, :]).astype(np.float32)
    motions_t = torch.from_numpy(motions).float()
    lengths_t = torch.from_numpy(lengths).long()

    device = torch.device("cpu")
    guo_eval = EvaluatorMDMWrapper("humanml", device)
    w_vectorizer = WordVectorizer(os.path.join(MDM_ROOT, "..", "text-to-motion", "glove"), "our_vab")
    MAX_TEXT_LEN = 20
    T2M_MEAN = np.load(os.path.join(MDM_ROOT, "dataset", "t2m_mean.npy"))
    T2M_STD = np.load(os.path.join(MDM_ROOT, "dataset", "t2m_std.npy"))

    def wrap_tokens(toks):
        if len(toks) < MAX_TEXT_LEN:
            wrapped = ["sos/OTHER"] + toks + ["eos/OTHER"]
            sent_len = len(wrapped)
            wrapped = wrapped + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
        else:
            wrapped = ["sos/OTHER"] + toks[:MAX_TEXT_LEN] + ["eos/OTHER"]
            sent_len = len(wrapped)
        return wrapped, sent_len

    print("Embedding text (Guo)...")
    guo_text_emb = []
    for toks in tokens_list:
        wrapped, sent_len = wrap_tokens(toks)
        we, po = [], []
        for tok in wrapped:
            w, p = w_vectorizer[tok]
            we.append(w[None, :]); po.append(p[None, :])
        word_embs = torch.from_numpy(np.concatenate(we, axis=0)[None, :]).float()
        pos_ohots = torch.from_numpy(np.concatenate(po, axis=0)[None, :]).float()
        cap_len_t = torch.tensor([sent_len])
        with torch.no_grad():
            emb = guo_eval.text_encoder(word_embs, pos_ohots, cap_len_t)
        guo_text_emb.append(emb[0].numpy())
    guo_text_emb = np.stack(guo_text_emb)

    print("Embedding motions (Guo, single-sort-invert, natural generation order)...")
    GUO_BATCH = 32
    guo_motion_emb = np.zeros((n_gen, 512), dtype=np.float32)
    n_full_batches = n_gen // GUO_BATCH
    for b in range(n_full_batches):
        idx = list(range(b * GUO_BATCH, (b + 1) * GUO_BATCH))
        m_raw = motions_t[idx].numpy()
        m = torch.from_numpy((m_raw - T2M_MEAN[None, None, :]) / T2M_STD[None, None, :]).float()
        l = lengths_t[idx]
        align_idx = np.argsort(l.numpy())[::-1].copy()
        inv = np.argsort(align_idx)
        with torch.no_grad():
            emb = guo_eval.get_motion_embeddings(m, l).numpy()[inv]
        for j, i in enumerate(idx):
            guo_motion_emb[i] = emb[j]

    n_usable = n_full_batches * GUO_BATCH
    print(f"embeddings ready, {n_usable} usable in natural batching")

    natural = r_precision_batches(guo_text_emb[:n_usable], guo_motion_emb[:n_usable], "euclidean")
    print("natural order top-1/2/3:", natural)

    rng = np.random.RandomState(0)
    trials = []
    for _ in range(my_args.n_trials):
        perm = rng.permutation(n_usable)
        r = r_precision_batches(guo_text_emb[perm], guo_motion_emb[perm], "euclidean")
        trials.append(r)
    trials = np.array(trials)
    print("shuffled trials top-3 mean/std:", trials[:, 2].mean(), trials[:, 2].std())

    sp_trials, ns_trials = [], []
    for _ in range(my_args.n_trials):
        perm = rng.permutation(n_usable)
        te, me = guo_text_emb[perm], guo_motion_emb[perm]
        perm_is_spatial = np.array([is_spatial(captions[i]) for i in perm])
        sp_pos, ns_pos = np.where(perm_is_spatial)[0], np.where(~perm_is_spatial)[0]
        sp_trials.append(r_precision_batches(te[sp_pos], me[sp_pos], "euclidean")[2])
        ns_trials.append(r_precision_batches(te[ns_pos], me[ns_pos], "euclidean")[2])
    print("shuffled spatial top-3 mean/std:", np.mean(sp_trials), np.std(sp_trials))
    print("shuffled non-spatial top-3 mean/std:", np.mean(ns_trials), np.std(ns_trials))
    print("shuffled gap (spatial-nonspatial) mean:", np.mean(sp_trials) - np.mean(ns_trials))

    result = {
        "n_usable": n_usable,
        "natural_order_top123": natural,
        "shuffled_overall_top3_trials": trials[:, 2].tolist(),
        "shuffled_spatial_top3_trials": sp_trials,
        "shuffled_nonspatial_top3_trials": ns_trials,
    }
    with open(my_args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
