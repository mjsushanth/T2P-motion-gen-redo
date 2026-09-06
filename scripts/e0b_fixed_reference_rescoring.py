"""E0b follow-up (per review SUP-20260906-28): fix a full-scale ground-truth reference (mu, cov)
and re-score the already-cached 128 generated motions against it. Zero regeneration cost -- no
diffusion sampling here, only encoding real ground-truth motions (fast) and re-encoding the
already-cached generated motions (fast) through the same evaluator used in E0b.

This removes the reference-redraw variance that SUP-25/26/27 showed was dominating E0b's FID
(a +30% swing from redrawing only the n=128 reference). The fixed reference here uses the full
test split (~4198 sequences, same scale as E0a), saved once so every future FID in this project
can be compared against the same fixed target.

Run from third_party/motion-diffusion-model/:
    PYTHONPATH=. python3 ../../scripts/e0b_fixed_reference_rescoring.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    sys.argv = ["x", "--model_path",
                "../../checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/model000475000.pt",
                "--eval_mode", "debug", "--dataset", "humanml"]

    from utils.parser_util import evaluation_parser
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance
    from data_loaders.get_data import get_dataset_loader

    args = evaluation_parser()
    device = torch.device("cpu")

    print("Building full-scale ground-truth reference (no generation, encoding only)...")
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None,
                                    split="test", hml_mode="gt")
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, device)

    gt_embeddings = []
    n_gt = 0
    for _, _, _, _, motions, m_lens, _ in gt_loader:
        with torch.no_grad():
            emb = eval_wrapper.get_motion_embeddings(motions=motions, m_lens=m_lens)
        gt_embeddings.append(emb.cpu().numpy())
        n_gt += motions.shape[0]
    gt_embeddings = np.concatenate(gt_embeddings, axis=0)
    print(f"Fixed ground-truth reference built from {n_gt} sequences, embedding shape {gt_embeddings.shape}")

    mu_ref, cov_ref = calculate_activation_statistics(gt_embeddings)
    ref_dir = Path("../../artifacts/e0/fixed_gt_reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ref_dir / "fixed_gt_reference.npz", mu=mu_ref, cov=cov_ref, n=n_gt,
             embeddings=gt_embeddings)
    print(f"Saved fixed reference (n={n_gt}) to {ref_dir / 'fixed_gt_reference.npz'} "
          "-- reusable for every future FID computation in this project.")

    print("Re-scoring the already-cached 128 generated motions against the fixed reference...")
    cache_dir = Path("../../artifacts/e0/e0b_generated_cache")
    gen_data = np.load(cache_dir / "generated_motions.npz")
    with open(cache_dir / "generated_meta.json") as f:
        gen_meta = json.load(f)
    n_gen = len(gen_meta)
    motions = np.stack([gen_data[f"motion_{i}"] for i in range(n_gen)])
    lengths = np.array([m["length"] for m in gen_meta])

    motions_t = torch.from_numpy(motions).float()
    lengths_t = torch.from_numpy(lengths).long()
    with torch.no_grad():
        gen_embeddings = eval_wrapper.get_motion_embeddings(motions=motions_t, m_lens=lengths_t).cpu().numpy()

    mu_gen, cov_gen = calculate_activation_statistics(gen_embeddings)
    fid_fixed_ref = float(calculate_frechet_distance(mu_gen, cov_gen, mu_ref, cov_ref))

    result = {
        "experiment": "E0b_fixed_reference_rescoring",
        "description": "FID of the SAME 128 cached generated motions (no regeneration) against "
                        "a FIXED full-scale (n={}) ground-truth reference, removing the "
                        "reference-redraw variance SUP-25/26/27 showed dominated the n=128 "
                        "vs n=128 comparisons in round 1/round 2.".format(n_gt),
        "n_ground_truth_reference": n_gt,
        "n_generated": n_gen,
        "fid_generated_vs_fixed_full_scale_reference": fid_fixed_ref,
        "reference_paper_fid": {"value": 0.544, "ci": 0.044, "tolerance_pct5": [0.5168, 0.5712]},
        "prior_measurements_for_comparison": {
            "round1_fid_n128_vs_n128": 1.0731,
            "round2_fid_n128_vs_n128": 1.3997,
            "note": "both rounds compared against a DIFFERENT small n=128 reference each time; "
                    "this measurement uses one FIXED, full-scale reference instead."
        }
    }
    out_path = "../../artifacts/e0/e0b_fixed_reference_rescoring_record.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
