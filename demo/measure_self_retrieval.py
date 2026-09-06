"""SUP-20260906-60/61: measures the real self-retrieval effect for the demo's new headline --
"a full caption finds its own motion N% of the time; truncated, it finds it M% of the time" --
using the EMBEDDING retriever (the same space R-Precision itself uses), not TF-IDF.

The director's own SUP-61 validated this premise with TF-IDF over HumanML3D's own tags (78.3%
full self-retrieval -> 55.0% truncated, over 300 sampled captions) specifically to check the
premise before building on it -- but flagged those numbers as establishing the premise, not as
display values, since the demo will use the embedding retriever, not TF-IDF, and the measured
0.145 R-Precision drop was itself measured in the evaluator's embedding space. This script
re-measures with that same embedding space, using HumanML3D's own real tags for truncation
(not spaCy) so this measurement is independent of SUP-57's separate spaCy-agreement question.
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
    retriever = EmbeddingRetriever()
    corpus_motion_ids = retriever.motion_ids  # aligned with retriever.embeddings rows
    corpus_embeddings = retriever.embeddings

    entries = _load_tagged_captions()
    random.seed(seed)
    # only sample captions whose motion actually made it into the retriever's corpus (motions
    # below MIN_MOTION_LEN were filtered out when the retriever built its embedding cache)
    valid_ids = set(corpus_motion_ids)
    entries = [e for e in entries if e[2] in valid_ids]
    sample = random.sample(entries, min(sample_size, len(entries)))

    def nearest_motion_id(emb):
        norms = np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(emb)
        norms[norms == 0] = 1e-8
        sims = (corpus_embeddings @ emb) / norms
        return corpus_motion_ids[int(sims.argmax())]

    n_total = 0
    n_shortened = 0
    n_self_full = 0
    n_self_trunc = 0
    n_different_retrieval = 0
    examples = []

    for caption, tokens, true_motion_id in sample:
        trunc_caption, trunc_tokens, used_fb, floored = truncate_tokens_first_action_clause(caption, tokens)
        shortened = trunc_caption.strip().lower() != caption.strip().lower()

        full_emb = retriever._embed_text(tokens)
        trunc_emb = retriever._embed_text(trunc_tokens)
        full_top1 = nearest_motion_id(full_emb)
        trunc_top1 = nearest_motion_id(trunc_emb)

        n_total += 1
        n_shortened += int(shortened)
        n_self_full += int(full_top1 == true_motion_id)
        n_self_trunc += int(trunc_top1 == true_motion_id)
        n_different_retrieval += int(full_top1 != trunc_top1)

        if len(examples) < 10 and full_top1 != trunc_top1:
            examples.append({
                "caption": caption, "truncated": trunc_caption, "true_motion_id": true_motion_id,
                "full_retrieval": full_top1, "truncated_retrieval": trunc_top1,
            })

    result = {
        "n_total": n_total,
        "pct_shortened": n_shortened / n_total,
        "pct_self_retrieval_full": n_self_full / n_total,
        "pct_self_retrieval_truncated": n_self_trunc / n_total,
        "self_retrieval_drop": (n_self_full - n_self_trunc) / n_total,
        "pct_different_retrieval": n_different_retrieval / n_total,
        "reference_tfidf_measurement_sup61": {
            "pct_shortened": 0.623, "pct_self_retrieval_full": 0.783,
            "pct_self_retrieval_truncated": 0.550, "pct_different_retrieval": 0.460,
            "note": "TF-IDF, HumanML3D's own tags, n=300 -- validated the premise, not a display value",
        },
        "sample_disagreement_examples": examples,
    }
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
