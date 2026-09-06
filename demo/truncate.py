"""Faithful reproduction of the original project's caption-truncation rule, adapted to run on
raw free-form user text instead of HumanML3D's pre-tagged corpus.

The original rule (read directly from the archived project, <ARCHIVE>/DL_T2P_IMPL.ipynb,
HumanML3DProcessor._filter_for_static_poses) operates on HumanML3D's own POS-tagged caption
format. This project's E1-pilot (scripts/e1_pilot_caption_truncation.py) ported that rule for
HumanML3D's own already-tagged captions. The demo needs the same rule applied to whatever a user
types, so this module adds a POS-tagging step (spaCy, en_core_web_sm) upstream, filtering
punctuation, to reproduce HumanML3D's own tagging convention (spaCy Universal Dependencies tags,
lowercased tokens, no punctuation tokens) closely enough for the same regex to fire the same way.
"""
import re

import spacy

CONJUNCTION_RE = re.compile(r"/CCONJ|/SCONJ|/ADV then|/ADV after|/ADV before")

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def pos_tag(caption: str) -> tuple[list[str], list[str]]:
    """Returns (words, tagged_tokens) with punctuation dropped, matching HumanML3D's own
    caption-tagging convention (e.g. 'a/DET person/NOUN walk/VERB ...')."""
    doc = _get_nlp()(caption.strip())
    words, tokens = [], []
    for tok in doc:
        if tok.is_punct or not tok.text.strip():
            continue
        words.append(tok.text.lower())
        tokens.append(f"{tok.text.lower()}/{tok.pos_}")
    return words, tokens


def truncate_first_action_clause(caption: str) -> tuple[str, bool]:
    """Applies the original project's own truncation rule to raw text.

    Returns (truncated_caption, used_fallback). used_fallback mirrors the diagnostic already
    established in the E1-pilot: True if the rule fell through to "first sentence" because no
    CCONJ/SCONJ tag was found (per docs/LANDMINES.md's note that the "/ADV then|after|before"
    literal-substring sub-branch likely never fires as intended)."""
    words, tokens = pos_tag(caption)
    if not words:
        return caption, True
    pos_text = " ".join(tokens)
    first_match = CONJUNCTION_RE.search(pos_text)
    if first_match is not None:
        pos_before_conj = pos_text[: first_match.start()].count(" ")
        trunc_words = words[:pos_before_conj]
        used_fallback = False
    else:
        sentences = re.split(r"[.!?]", caption)
        first_sentence = sentences[0].strip() if sentences else caption
        n = len(pos_tag(first_sentence)[0])
        trunc_words = words[:n]
        used_fallback = True
    if len(trunc_words) < 3:
        trunc_words = words[:3]
    return " ".join(trunc_words), used_fallback
