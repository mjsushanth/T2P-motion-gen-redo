"""Bootstrap error bar on E3's overall generated-vs-real FID, requested by peer review
(SUP-20260908-13): a single FID with no replication variance cannot be compared to MDM's
published 0.544+-.044 (a 20-replication figure) without one. No regeneration needed -- embeds
the same cached batches scripts/e3_generate.py produced (once) via the Guo evaluator, then
resamples both the generated and ground-truth embedding pools with replacement at their own
n=4,640 to get a distribution of FID values.

This does NOT attempt to close docs/DECISIONS.md D-03's FID half on its own: bootstrap variance
describes stability under resampling of this project's own two fixed embedding pools, not the
replication variance MDM's own protocol measures (independent generations/reference draws, which
this project's single generation run cannot provide after the fact). It answers a narrower,
still useful question -- is E3's own FID a well-determined number given its own data, or does it
swing wildly under resampling -- and is reported as exactly that, not as a substitute for the
paper's own replication protocol.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    PYTHONPATH=. python3 ../../scripts/e3_fid_bootstrap.py --gen-dir ../../artifacts/e3 \
        --out-json ../../artifacts/e3/fid_bootstrap.json --n-trials 1000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_score import load_generated_batches, embed_ground_truth_guo  # noqa: E402


def embed_generated_guo(guo_eval, motions_t, lengths_t, batch_size=32):
    n = motions_t.shape[0]
    embs = np.zeros((n, 512), dtype=np.float32)
    n_full = n // batch_size
    for b in range(n_full):
        idx = list(range(b * batch_size, (b + 1) * batch_size))
        m = motions_t[idx]
        l = lengths_t[idx]
        align_idx = np.argsort(l.numpy())[::-1].copy()
        inv = np.argsort(align_idx)
        with torch.no_grad():
            emb = guo_eval.get_motion_embeddings(m, l).numpy()[inv]
        for j, i in enumerate(idx):
            embs[i] = emb[j]
    used = n_full * batch_size
    return embs[:used]


def bootstrap_fid(gen_emb, real_emb, n_trials, rng):
    from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance
    n_gen, n_real = gen_emb.shape[0], real_emb.shape[0]
    point_mu_g, point_cov_g = calculate_activation_statistics(gen_emb)
    point_mu_r, point_cov_r = calculate_activation_statistics(real_emb)
    point_fid = float(calculate_frechet_distance(point_mu_g, point_cov_g, point_mu_r, point_cov_r))

    fids = []
    for _ in range(n_trials):
        gi = rng.choice(n_gen, size=n_gen, replace=True)
        ri = rng.choice(n_real, size=n_real, replace=True)
        mu_g, cov_g = calculate_activation_statistics(gen_emb[gi])
        mu_r, cov_r = calculate_activation_statistics(real_emb[ri])
        fids.append(calculate_frechet_distance(mu_g, cov_g, mu_r, cov_r))
    fids = np.array(fids)
    return {
        "point_estimate_fid": point_fid,
        "n_gen": int(n_gen),
        "n_real": int(n_real),
        "n_trials": n_trials,
        "bootstrap_mean": float(fids.mean()),
        "bootstrap_std": float(fids.std()),
        "bootstrap_ci95": [float(np.percentile(fids, 2.5)), float(np.percentile(fids, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

    print("Loading cached generated batches...")
    motions_raw, lengths, captions, tokens_list, cap_lens = load_generated_batches(args.gen_dir)
    n_gen = len(captions)
    print(f"{n_gen} generated samples loaded, motion shape {motions_raw.shape}")

    import os
    MDM_ROOT = os.path.abspath(os.getcwd())
    mdm_mean = np.load(os.path.join(MDM_ROOT, "dataset", "HumanML3D", "Mean.npy"))
    mdm_std = np.load(os.path.join(MDM_ROOT, "dataset", "HumanML3D", "Std.npy"))
    motions = (motions_raw * mdm_std[None, None, :] + mdm_mean[None, None, :]).astype(np.float32)
    T2M_MEAN = np.load(os.path.join(MDM_ROOT, "dataset", "t2m_mean.npy"))
    T2M_STD = np.load(os.path.join(MDM_ROOT, "dataset", "t2m_std.npy"))
    motions_norm = (motions - T2M_MEAN[None, None, :]) / T2M_STD[None, None, :]
    motions_t = torch.from_numpy(motions_norm.astype(np.float32))
    lengths_t = torch.from_numpy(lengths).long()

    device = torch.device("cpu")
    guo_eval = EvaluatorMDMWrapper("humanml", device)

    print("Embedding generated motions (Guo, batches of 32, single-sort-and-invert)...")
    gen_emb = embed_generated_guo(guo_eval, motions_t, lengths_t)
    print(f"Generated embeddings: {gen_emb.shape}")

    print("Embedding ground-truth motions (Guo) for FID...")
    gt_emb, gt_captions = embed_ground_truth_guo(guo_eval)
    print(f"Ground-truth embeddings: {gt_emb.shape}")

    rng = np.random.default_rng(args.seed)
    print(f"Bootstrapping FID ({args.n_trials} trials, resampling both sides with replacement)...")
    result = bootstrap_fid(gen_emb, gt_emb, args.n_trials, rng)
    result["seed"] = args.seed
    print(json.dumps(result, indent=2))

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
