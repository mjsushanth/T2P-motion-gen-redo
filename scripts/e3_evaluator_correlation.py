"""Per-sample Guo/TMR correlation on E3's own full-scale generated set (n=4,640).

Why this exists: the deck's slide 9 (planner spec, deck/PRESENTER_NOTES.md) cites
"r = 0.328 per-sample correlation between the two evaluators" as licensing the reading that
Guo/TMR's opposite-sign spatial-vs-non-spatial effect (E3, docs/EXPERIMENT_DESIGN_E3.md S9.1)
is instrument-dependence rather than noise. That r=0.328 figure, verified against its actual
source (notebooks/02_tmr_second_evaluator.ipynb, cell 055f57c3), was computed on n_gen=128 --
E0b's cached generation, the same small sample E3 S9.5 later showed was NOT representative of
this checkpoint at scale (its own R-Precision-top3, 0.7578, was a 3.71 sigma outlier from E3's
n=4,640 figure of 0.6172). Citing an n=128 correlation as evidence about an n=4,640 experiment
is the same class of small-sample-generalization error this project has already caught and
corrected twice elsewhere (the superseded 0.7578 figure itself, and the FID rough-floor
denominator bug) -- so this script computes the real thing: per-sample cosine agreement between
Guo and TMR on E3's own 4,640 generated motions, overall and split spatial/non-spatial, reusing
the exact embedding conventions scripts/e3_score.py already validated (denormalization,
single-sort-and-invert for Guo, TMR's own normalizer/collate convention) rather than
reimplementing them, and no regeneration -- same cached batches, embedded fresh (embeddings
themselves are not cached anywhere by e3_score.py).

Run from third_party/motion-diffusion-model/:
    cd third_party/motion-diffusion-model
    PYTHONPATH=. python3 ../../scripts/e3_evaluator_correlation.py \
        --gen-dir ../../artifacts/e3 --out-json ../../artifacts/e3/evaluator_correlation.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_score import load_generated_batches, is_spatial  # noqa: E402


def cosine_score_01(a, b):
    a_n = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (np.sum(a_n * b_n, axis=-1)) / 2 + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    MDM_ROOT = os.path.abspath(os.getcwd())
    TMR_ROOT = os.path.join(MDM_ROOT, "..", "TMR")
    sys.path.insert(0, MDM_ROOT)

    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

    print("Loading cached generated batches...")
    motions_raw, lengths, captions, tokens_list, cap_lens = load_generated_batches(args.gen_dir)
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
    used_n = n_full_batches * GUO_BATCH
    print(f"Guo embeddings ready ({n_full_batches} full batches, {n_gen - used_n} leftover excluded)")

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

    print(f"Embedding motions + text (TMR), {used_n} samples (matching Guo's usable count)...")
    tmr_motion_lat, tmr_text_lat = [], []
    for i in range(used_n):
        mt = tmr_normalizer(torch.from_numpy(motions[i]).float())
        x_dict = collate_x_dict([{"x": mt, "length": len(mt)}])
        with torch.inference_mode():
            tmr_motion_lat.append(tmr_model.encode(x_dict, sample_mean=True)[0].numpy())
        x_dict_t = collate_x_dict(tmr_text_model([captions[i]]))
        with torch.inference_mode():
            tmr_text_lat.append(tmr_model.encode(x_dict_t, sample_mean=True)[0].numpy())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{used_n} TMR-embedded")
    tmr_motion_lat = np.stack(tmr_motion_lat)
    tmr_text_lat = np.stack(tmr_text_lat)

    guo_scores = cosine_score_01(guo_text_emb[:used_n], guo_motion_emb[:used_n])
    tmr_scores = cosine_score_01(tmr_text_lat, tmr_motion_lat)

    spatial_idx = np.array([i for i in range(used_n) if is_spatial(captions[i])])
    nonspatial_idx = np.array([i for i in range(used_n) if not is_spatial(captions[i])])

    def corr(idx):
        if len(idx) < 2:
            return None
        return float(np.corrcoef(guo_scores[idx], tmr_scores[idx])[0, 1])

    result = {
        "n_used": used_n,
        "n_spatial": int(len(spatial_idx)),
        "n_nonspatial": int(len(nonspatial_idx)),
        "pearson_r_overall": corr(np.arange(used_n)),
        "pearson_r_spatial": corr(spatial_idx),
        "pearson_r_nonspatial": corr(nonspatial_idx),
        "note": (
            "Full-scale (n=4,640-batch-usable) equivalent of notebooks/02's own n=128 "
            "Guo/TMR per-sample correlation (r=0.328 there). Computed for slide 9 verification "
            "-- see scripts/e3_evaluator_correlation.py docstring for why the n=128 figure "
            "should not be cited for an n=4,640 claim without checking this first."
        ),
    }
    print(json.dumps(result, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
