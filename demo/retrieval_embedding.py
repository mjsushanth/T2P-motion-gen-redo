"""TF-IDF alone is a weak "just look it up" baseline -- the strongest honest
form of retrieval uses the same validated joint text-motion embedding space this project's whole
evaluation harness already relies on (EvaluatorMDMWrapper, ground-truth R-Precision reproduced to
within 0.0036 of a published reference across four independent full-split runs). Beating TF-IDF
and getting "you used a weak lookup" as the reply is not good enough; beating this retriever
leaves no such reply.

This is TEXT-TO-MOTION retrieval (embed the query caption, compare against real corpus MOTION
embeddings), not text-to-caption retrieval -- matching exactly what R-Precision itself measures,
and what an earlier caption-to-caption version of this file got wrong: that version embedded
captions on both sides through a text encoder trained for cross-modal (text-motion) alignment,
not for text-to-text discrimination, and it showed (semantically poor top-1 matches, spuriously
uniform ~0.98+ similarities regardless of query) -- caught by inspecting real query results
before trusting the approach, not assumed to work because the component was well-validated
elsewhere. TF-IDF is kept (retrieval.py) as a labelled, cheaper floor, not replaced.
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch

MDM_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model")
CORPUS_TEXT_DIR = os.path.join(MDM_ROOT, "dataset", "HumanML3D", "texts")
CORPUS_MOTION_DIR = os.path.join(MDM_ROOT, "dataset", "HumanML3D", "new_joint_vecs")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "demo", "embedding_retrieval_cache.pkl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncate import pos_tag as spacy_pos_tag  # noqa: E402

MAX_TEXT_LEN = 20
MAX_MOTION_LEN = 196
MIN_MOTION_LEN = 40


def _load_corpus_motion_ids():
    """One representative caption + motion_id per real materialized motion file."""
    entries = []
    for path in sorted(glob.glob(os.path.join(CORPUS_TEXT_DIR, "*.txt"))):
        motion_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or "#" not in line:
                    continue
                caption = line.split("#")[0].strip()
                if caption:
                    entries.append((caption, motion_id))
                    break  # one caption per motion is enough to label a retrieval hit
    return entries


def _wrap_tokens(tokens: list[str]) -> tuple[list[str], int]:
    """Same sos/eos/unk padding convention as Text2MotionDatasetV2.__getitem__."""
    if len(tokens) < MAX_TEXT_LEN:
        wrapped = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(wrapped)
        wrapped = wrapped + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        wrapped = tokens[:MAX_TEXT_LEN]
        wrapped = ["sos/OTHER"] + wrapped + ["eos/OTHER"]
        sent_len = len(wrapped)
    return wrapped, sent_len


class EmbeddingRetriever:
    def __init__(self, device="cpu"):
        sys.path.insert(0, os.path.abspath(MDM_ROOT))
        from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
        from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

        self.device = device
        old_cwd = os.getcwd()
        try:
            os.chdir(MDM_ROOT)
            self.w_vectorizer = WordVectorizer("glove", "our_vab")
            self.evaluator = EvaluatorMDMWrapper("humanml", device)
            self.t2m_mean = np.load(os.path.join("dataset", "t2m_mean.npy"))
            self.t2m_std = np.load(os.path.join("dataset", "t2m_std.npy"))
        finally:
            os.chdir(old_cwd)

        self.captions, self.motion_ids, self.embeddings = self._load_or_build_corpus_embeddings()

    def _embed_text(self, tokens: list[str]) -> np.ndarray:
        wrapped, sent_len = _wrap_tokens(tokens)
        we, po = [], []
        for tok in wrapped:
            w, p = self.w_vectorizer[tok]
            we.append(w[None, :])
            po.append(p[None, :])
        word_embs = torch.from_numpy(np.concatenate(we, axis=0)[None, :]).float()
        pos_ohots = torch.from_numpy(np.concatenate(po, axis=0)[None, :]).float()
        cap_lens = torch.tensor([sent_len])
        with torch.no_grad():
            emb = self.evaluator.text_encoder(word_embs, pos_ohots, cap_lens)
        return emb[0].cpu().numpy()

    def _embed_motions_batch(self, motion_ids: list[str]) -> np.ndarray:
        motions, lengths = [], []
        for mid in motion_ids:
            raw = np.load(os.path.join(CORPUS_MOTION_DIR, f"{mid}.npy"))
            m_len = min(raw.shape[0], MAX_MOTION_LEN)
            normed = (raw[:m_len] - self.t2m_mean) / self.t2m_std
            if m_len < MAX_MOTION_LEN:
                normed = np.concatenate(
                    [normed, np.zeros((MAX_MOTION_LEN - m_len, normed.shape[1]))], axis=0
                )
            motions.append(normed[None, :])
            lengths.append(m_len)
        motions_t = torch.from_numpy(np.concatenate(motions, axis=0)).float()
        lengths_t = torch.tensor(lengths)

        # get_motion_embeddings reorders by descending length internally and does NOT undo it
        # ("Please note that the results does not following the order of inputs") -- invert here.
        order = np.argsort(-lengths_t.numpy())
        inv_order = np.argsort(order)
        emb = self.evaluator.get_motion_embeddings(motions_t[order], lengths_t[order])
        return emb[inv_order].cpu().numpy()

    def _load_or_build_corpus_embeddings(self):
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("version") == 2:
                return cached["captions"], cached["motion_ids"], cached["embeddings"]

        entries = _load_corpus_motion_ids()
        captions = [c for c, _ in entries]
        motion_ids = [m for _, m in entries]
        print(f"embedding {len(motion_ids)} corpus MOTIONS (one-time, cached after)...")

        batch_size = 64
        all_embs = []
        kept_captions, kept_ids = [], []
        for i in range(0, len(motion_ids), batch_size):
            batch_ids = motion_ids[i:i + batch_size]
            batch_caps = captions[i:i + batch_size]
            # skip motions shorter than MIN_MOTION_LEN, matching Text2MotionDatasetV2's own filter
            valid = []
            for mid, cap in zip(batch_ids, batch_caps):
                path = os.path.join(CORPUS_MOTION_DIR, f"{mid}.npy")
                if os.path.exists(path) and np.load(path).shape[0] >= MIN_MOTION_LEN:
                    valid.append((mid, cap))
            if not valid:
                continue
            v_ids = [v[0] for v in valid]
            v_caps = [v[1] for v in valid]
            all_embs.append(self._embed_motions_batch(v_ids))
            kept_ids.extend(v_ids)
            kept_captions.extend(v_caps)

        embeddings = np.concatenate(all_embs, axis=0)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                {"version": 2, "captions": kept_captions, "motion_ids": kept_ids, "embeddings": embeddings}, f
            )
        return kept_captions, kept_ids, embeddings

    def nearest(self, query: str, k: int = 1):
        _, query_tokens = spacy_pos_tag(query)
        query_emb = self._embed_text(query_tokens)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        norms[norms == 0] = 1e-8
        sims = (self.embeddings @ query_emb) / norms
        top_idx = sims.argsort()[::-1][:k]
        return [(self.captions[i], self.motion_ids[i], float(sims[i])) for i in top_idx]
