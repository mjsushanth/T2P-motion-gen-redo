"""The demo's truncation (demo/truncate.py, spaCy POS-tagging on free text)
must be validated against the truncation actually measured in the E1-pilot
(scripts/e1_pilot_caption_truncation.py, HumanML3D's own pre-tagged captions) -- otherwise the
demo could show a rule nobody measured. Runs both over the same real HumanML3D captions and
reports the agreement rate.

Comparison is done punctuation-normalized (lowercased, punctuation stripped, whitespace
collapsed): the two methods differ in how they split on punctuation that HumanML3D's raw
captions sometimes glue directly to a word with no space (e.g. "side ,walks"), which is a
tokenization-surface difference, not a difference in WHERE the rule decides to cut -- comparing
raw strings would conflate the two.
"""
import glob
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from e1_pilot_caption_truncation import truncate_tokens_first_action_clause as tag_based_truncate
from truncate import truncate_first_action_clause as spacy_based_truncate

CORPUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "third_party", "motion-diffusion-model",
    "dataset", "HumanML3D", "texts",
)


def _normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def main(sample_size: int = 3000, seed: int = 10):
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    random.seed(seed)
    sample_files = random.sample(files, min(sample_size, len(files)))

    n_total = 0
    n_agree_raw = 0
    n_agree_norm = 0
    n_fallback_agree = 0
    disagreements = []
    for path in sample_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or "#" not in line:
                    continue
                parts = line.split("#")
                caption = parts[0].strip()
                tokens = parts[1].split(" ") if len(parts) > 1 else []
                if not caption or not tokens:
                    continue
                tag_trunc, _, tag_fb, _ = tag_based_truncate(caption, tokens)
                spacy_trunc, spacy_fb = spacy_based_truncate(caption)
                n_total += 1
                if tag_trunc.strip().lower() == spacy_trunc.strip().lower():
                    n_agree_raw += 1
                a, b = _normalize(tag_trunc), _normalize(spacy_trunc)
                if a == b:
                    n_agree_norm += 1
                else:
                    disagreements.append((caption, tag_trunc, spacy_trunc, tag_fb, spacy_fb))
                if tag_fb == spacy_fb:
                    n_fallback_agree += 1

    result = {
        "n_total": n_total,
        "n_agree_raw": n_agree_raw,
        "pct_agree_raw": n_agree_raw / n_total,
        "n_agree_punct_normalized": n_agree_norm,
        "pct_agree_punct_normalized": n_agree_norm / n_total,
        "n_fallback_flag_agree": n_fallback_agree,
        "pct_fallback_flag_agree": n_fallback_agree / n_total,
        "sample_disagreements": disagreements[:15],
    }
    return result


if __name__ == "__main__":
    import json

    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "sample_disagreements"}, indent=2))
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "demo", "truncation_agreement_record.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print("Saved:", out_path)
