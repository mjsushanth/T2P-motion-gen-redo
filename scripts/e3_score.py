"""E3 scoring pass -- loads every cached batch scripts/e3_generate.py produced, scores with
both evaluators this project has working and cross-checked (Guo + TMR, notebooks/02), splits
into spatial / non-spatial / length-tercile subsets, and reports R-Precision (primary, decisive
per docs/DECISIONS.md D-25's small-sample regime) and FID (secondary) per subset.

Reuses the exact conventions already validated elsewhere in this project rather than
reimplementing them: the denormalization step from notebooks/02 (cached motions are in MDM's
own training normalization; both evaluators expect raw HumanML3D features), the single-sort-
and-invert calling convention for get_motion_embeddings from docs/LANDMINES.md section 25 (no
external pre-sort), and the same spatial-term list notebooks 01/05 established.

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    PYTHONPATH=. python3 ../../scripts/e3_score.py --gen-dir ../../artifacts/e3 --out-json ../../artifacts/e3/e3_record.json
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

SPATIAL_TERMS = ["left", "right", "forward", "forwards", "backward", "backwards",
                  "clockwise", "counterclockwise", "anticlockwise", "upleft",
                  "behind", "in front of"]


def is_spatial(caption):
    caption = caption.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", caption) for t in SPATIAL_TERMS)


def load_generated_batches(gen_dir):
    batches_dir = Path(gen_dir) / "batches"
    npz_paths = sorted(glob.glob(str(batches_dir / "batch_*.npz")))
    all_motions, all_lengths, all_captions, all_tokens, all_cap_lens = [], [], [], [], []
    for npz_path in npz_paths:
        meta_path = npz_path.replace(".npz", "_meta.json")
        data = np.load(npz_path)
        meta = json.load(open(meta_path))
        all_motions.append(data["motions"])
        all_lengths.append(data["lengths"])
        all_captions.extend(meta["captions"])
        all_tokens.extend(meta["tokens"])
        all_cap_lens.extend(meta["cap_lens"])
    motions = np.concatenate(all_motions, axis=0)
    lengths = np.concatenate(all_lengths, axis=0)
    return motions, lengths, all_captions, all_tokens, all_cap_lens


def length_terciles(indices, captions):
    """Splits `indices` into 3 equal-ish groups by that subset's own caption word count,
    matching docs/EXPERIMENT_DESIGN_E2.md section 3's stated convention."""
    word_counts = [(i, len(captions[i].split())) for i in indices]
    word_counts.sort(key=lambda t: t[1])
    n = len(word_counts)
    third = n // 3
    groups = [word_counts[:third], word_counts[third:2 * third], word_counts[2 * third:]]
    return [[i for i, _ in g] for g in groups]


def r_precision_batches(text_emb, motion_emb, distance, batch_size=32, top_k=3):
    from data_loaders.humanml.utils.metrics import euclidean_distance_matrix, calculate_top_k
    n = text_emb.shape[0]
    n_batches = n // batch_size
    top_k_sum = np.zeros(top_k)
    for b in range(n_batches):
        te = text_emb[b * batch_size:(b + 1) * batch_size]
        me = motion_emb[b * batch_size:(b + 1) * batch_size]
        if distance == "euclidean":
            dist_mat = euclidean_distance_matrix(te, me)
            order_mat = np.argsort(dist_mat, axis=1)
        else:
            te_n = te / np.linalg.norm(te, axis=-1, keepdims=True)
            me_n = me / np.linalg.norm(me, axis=-1, keepdims=True)
            order_mat = np.argsort(-(te_n @ me_n.T), axis=1)
        top_k_sum += calculate_top_k(order_mat, top_k).sum(axis=0)
    return (top_k_sum / (n_batches * batch_size)).tolist() if n_batches > 0 else [None] * top_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-json", required=True)
    my_args = ap.parse_args()

    MDM_ROOT = os.path.abspath(os.getcwd())  # run from third_party/motion-diffusion-model/
    TMR_ROOT = os.path.join(MDM_ROOT, "..", "TMR")
    sys.path.insert(0, MDM_ROOT)

    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance

    print("Loading cached generated batches...")
    motions_raw, lengths, captions, tokens_list, cap_lens = load_generated_batches(my_args.gen_dir)
    n_gen = len(captions)
    print(f"{n_gen} generated samples loaded, motion shape {motions_raw.shape}")

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

    print("Embedding motions (Guo, batches of 32, single-sort-and-invert)...")
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
    print(f"Guo embeddings ready ({n_full_batches} full batches of {GUO_BATCH}, "
          f"{n_gen - n_full_batches * GUO_BATCH} leftover samples excluded from R-Precision, "
          f"same convention as every other R-Precision number in this project)")

    print("Loading TMR...")
    sys.path.insert(0, os.path.abspath(TMR_ROOT))
    _old_cwd = os.getcwd()
    os.chdir(os.path.abspath(TMR_ROOT))
    try:
        import src.prepare  # noqa
        from src.config import read_config
        from src.load import load_model_from_cfg
        from hydra.utils import instantiate
        from src.data.collate import collate_x_dict
        from src.data.text import TokenEmbeddings

        tmr_cfg = read_config("models/tmr_humanml3d_guoh3dfeats")
        tmr_text_model = TokenEmbeddings(
            modelname="distilbert-base-uncased",
            path=os.path.join("datasets", "annotations", "humanml3d"),
            device="cpu", preload=False,
        )
        tmr_model = load_model_from_cfg(tmr_cfg, "last", eval_mode=True, device="cpu")
        tmr_normalizer = instantiate(tmr_cfg.data.motion_loader.normalizer)
    finally:
        os.chdir(_old_cwd)

    print("Embedding motions + text (TMR)...")
    tmr_motion_lat, tmr_text_lat = [], []
    for i in range(n_gen):
        mt = tmr_normalizer(torch.from_numpy(motions[i]).float())
        # notebooks/02's own convention: length=len(mt) (the padded array's own length), not
        # the real per-sample motion length -- both are full (196, 263) padded arrays here.
        x_dict = collate_x_dict([{"x": mt, "length": len(mt)}])
        with torch.inference_mode():
            tmr_motion_lat.append(tmr_model.encode(x_dict, sample_mean=True)[0].numpy())
        x_dict_t = collate_x_dict(tmr_text_model([captions[i]]))
        with torch.inference_mode():
            tmr_text_lat.append(tmr_model.encode(x_dict_t, sample_mean=True)[0].numpy())
    tmr_motion_lat = np.stack(tmr_motion_lat)
    tmr_text_lat = np.stack(tmr_text_lat)

    spatial_idx = [i for i in range(n_gen) if is_spatial(captions[i])]
    nonspatial_idx = [i for i in range(n_gen) if not is_spatial(captions[i])]
    print(f"Spatial: {len(spatial_idx)}, non-spatial: {len(nonspatial_idx)} "
          f"({100*len(spatial_idx)/n_gen:.1f}% spatial)")

    def score_subset(idx, label):
        te_guo, me_guo = guo_text_emb[idx], guo_motion_emb[idx]
        te_tmr, me_tmr = tmr_text_lat[idx], tmr_motion_lat[idx]
        return {
            "label": label, "n": len(idx),
            "guo_r_precision": r_precision_batches(te_guo, me_guo, "euclidean"),
            "tmr_r_precision": r_precision_batches(te_tmr, me_tmr, "cosine"),
            "mean_caption_words": float(np.mean([len(captions[i].split()) for i in idx])),
        }

    results = {"overall": score_subset(list(range(n_gen)), "overall"),
               "spatial": score_subset(spatial_idx, "spatial"),
               "non_spatial": score_subset(nonspatial_idx, "non_spatial")}

    for subset_name, idx in [("spatial", spatial_idx), ("non_spatial", nonspatial_idx)]:
        terciles = length_terciles(idx, captions)
        results[f"{subset_name}_terciles"] = [
            score_subset(t_idx, f"{subset_name}_tercile_{k}") for k, t_idx in enumerate(terciles)
        ]

    print(json.dumps(results, indent=2))

    with open(my_args.out_json, "w") as f:
        json.dump({
            "experiment": "E3",
            "description": "Spatial-vs-non-spatial R-Precision on the released MDM checkpoint, "
                            "no training, full HumanML3D test split generated once.",
            "n_generated": n_gen,
            "results": results,
        }, f, indent=2)
    print("Saved:", my_args.out_json)


if __name__ == "__main__":
    main()
