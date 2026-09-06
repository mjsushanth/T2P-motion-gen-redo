"""Nearest-neighbour retrieval baseline, per SUP-20260906-43: "just look it up" is what a
sceptical viewer silently thinks when shown a generated motion; showing the actual nearest
real-motion match answers that objection directly instead of ducking it.

Deliberately simple (TF-IDF cosine similarity over the materialized corpus's real captions, not
a learned embedding) -- the point is to show a legitimate, unglamorous baseline a viewer can
trust is not tuned to look good, not to build the strongest possible retrieval system.
"""
import glob
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model",
    "dataset", "HumanML3D", "texts",
)


def _load_corpus():
    captions, motion_ids = [], []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt"))):
        motion_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or "#" not in line:
                    continue
                caption = line.split("#")[0].strip()
                if caption:
                    captions.append(caption)
                    motion_ids.append(motion_id)
    return captions, motion_ids


class NearestNeighborRetriever:
    def __init__(self):
        self.captions, self.motion_ids = _load_corpus()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.captions)

    def nearest(self, query: str, k: int = 1):
        """Returns a list of (caption, motion_id, similarity) for the top-k nearest real
        captions in the materialized corpus, by TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [(self.captions[i], self.motion_ids[i], float(sims[i])) for i in top_idx]
