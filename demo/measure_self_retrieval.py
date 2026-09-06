"""SUP-20260906-60/61/62: measures the real self-retrieval effect for the demo's new headline --
"a full caption finds its own motion N% of the time; truncated, it finds it M% of the time" --
with BOTH retrievers (TF-IDF and the embedding retriever) over the SAME sample, so which one
leads the demo is chosen on stated, measured grounds, not on whichever looked better.

The director's own SUP-61 measured this premise with TF-IDF over HumanML3D's own tags (78.3%
full self-retrieval -> 55.0% truncated, n=300) specifically to check the premise before building
on it. This script independently re-verifies that TF-IDF number (not just quoting it) and
compares it against the embedding retriever on the identical sample, using HumanML3D's own real
tags for truncation (not spaCy) so this measurement is independent of SUP-57's separate
spaCy-agreement question.
"""
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from retrieval_embedding import EmbeddingRetriever, CORPUS_TEXT_DIR  # noqa: E402
from retrieval import NearestNeighborRetriever  # noqa: E402
from e1_pilot_caption_truncation import truncate_tokens_first_action_clause  # noqa: E402


def _load_tagged_captions():
    """(caption, tokens, motion_id) triples using HumanML3D's own real tags -- one caption per
    line, all lines per motion (a motion can have multiple captions)."""
    entries = []
    for path in sorted(glob.glob(os.path.join(CORPUS_TEXT_DIR, "*.txt"))):
        motion_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or "#" not in line:
                    continue
                parts = line.split("#")
                caption = parts[0].strip()
                tokens = parts[1].split(" ") if len(parts) > 1 else []
                if caption and tokens:
                    entries.append((caption, tokens, motion_id))
    return entries


def main(sample_size: int = 300, seed: int = 10):
    emb_retriever = EmbeddingRetriever()
    corpus_motion_ids = emb_retriever.motion_ids  # aligned with emb_retriever.embeddings rows
    corpus_embeddings = emb_retriever.embeddings
    tfidf_retriever = NearestNeighborRetriever()

    entries = _load_tagged_captions()
    random.seed(seed)
    # only sample captions whose motion actually made it into the embedding retriever's corpus
    # (motions below MIN_MOTION_LEN were filtered out when its embedding cache was built)
    valid_ids = set(corpus_motion_ids)
    entries = [e for e in entries if e[2] in valid_ids]
    sample = random.sample(entries, min(sample_size, len(entries)))

    def nearest_motion_id_embedding(emb):
        norms = np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(emb)
        norms[norms == 0] = 1e-8
        sims = (corpus_embeddings @ emb) / norms
        return corpus_motion_ids[int(sims.argmax())]

    def nearest_motion_id_tfidf(caption_text):
        _, motion_id, _ = tfidf_retriever.nearest(caption_text, k=1)[0]
        return motion_id

    stats = {method: {"n_shortened": 0, "n_self_full": 0, "n_self_trunc": 0, "n_diff": 0}
              for method in ("embedding", "tfidf")}
    n_total = 0
    examples = {"embedding": [], "tfidf": []}

    for caption, tokens, true_motion_id in sample:
        trunc_caption, trunc_tokens, used_fb, floored = truncate_tokens_first_action_clause(caption, tokens)
        shortened = trunc_caption.strip().lower() != caption.strip().lower()
        n_total += 1

        # embedding retriever
        full_emb = emb_retriever._embed_text(tokens)
        trunc_emb = emb_retriever._embed_text(trunc_tokens)
        e_full = nearest_motion_id_embedding(full_emb)
        e_trunc = nearest_motion_id_embedding(trunc_emb)
        s = stats["embedding"]
        s["n_shortened"] += int(shortened)
        s["n_self_full"] += int(e_full == true_motion_id)
        s["n_self_trunc"] += int(e_trunc == true_motion_id)
        s["n_diff"] += int(e_full != e_trunc)
        if len(examples["embedding"]) < 10 and e_full != e_trunc:
            examples["embedding"].append({"caption": caption, "truncated": trunc_caption,
                                            "true_motion_id": true_motion_id,
                                            "full_retrieval": e_full, "truncated_retrieval": e_trunc})

        # TF-IDF retriever (caption-text based, same truncated/full text)
        t_full = nearest_motion_id_tfidf(caption)
        t_trunc = nearest_motion_id_tfidf(trunc_caption)
        s = stats["tfidf"]
        s["n_shortened"] += int(shortened)
        s["n_self_full"] += int(t_full == true_motion_id)
        s["n_self_trunc"] += int(t_trunc == true_motion_id)
        s["n_diff"] += int(t_full != t_trunc)
        if len(examples["tfidf"]) < 10 and t_full != t_trunc:
            examples["tfidf"].append({"caption": caption, "truncated": trunc_caption,
                                        "true_motion_id": true_motion_id,
                                        "full_retrieval": t_full, "truncated_retrieval": t_trunc})

    result = {"n_total": n_total}
    for method in ("embedding", "tfidf"):
        s = stats[method]
        result[method] = {
            "pct_shortened": s["n_shortened"] / n_total,
            "pct_self_retrieval_full": s["n_self_full"] / n_total,
            "pct_self_retrieval_truncated": s["n_self_trunc"] / n_total,
            "self_retrieval_drop": (s["n_self_full"] - s["n_self_trunc"]) / n_total,
            "pct_different_retrieval": s["n_diff"] / n_total,
        }
    result["sample_disagreement_examples"] = examples
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "sample_disagreement_examples"}, indent=2))
    print()
    print("--- example retrieval changes ---")
    for ex in r["sample_disagreement_examples"]:
        print(ex)
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "demo", "self_retrieval_record.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print("Saved:", out_path)
