"""Helper functions available to generated attention-prediction code.

These are injected into the execution environment so generated code can import
them via `from helpers import *`.
"""

import numpy as np
import spacy

_nlp = None
_gpt2_tok = None


def get_nlp():
    """Return a cached spacy English model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_gpt2_tokenizer():
    """Return a cached GPT2 tokenizer."""
    global _gpt2_tok
    if _gpt2_tok is None:
        from transformers import GPT2Tokenizer
        _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    return _gpt2_tok


def gpt2_tokenize(sentence: str) -> list[str]:
    """Tokenize a sentence using GPT2 BPE tokenizer.

    Returns a list of token strings. Leading spaces are included in tokens
    (e.g. " cat" not "cat") to match GPT2's convention.
    """
    tok = _get_gpt2_tokenizer()
    ids = tok.encode(sentence)
    return [tok.decode([i]) for i in ids]


def spacy_parse(sentence: str):
    """Parse a sentence with spacy, returning a Doc object."""
    return get_nlp()(sentence)


def align_spacy_to_gpt2(sentence: str) -> list[list[int]]:
    """For each spacy token, return the list of overlapping GPT2 token indices.

    Uses character offsets to align between the two tokenizations.
    """
    doc = spacy_parse(sentence)
    gpt2_tokens = gpt2_tokenize(sentence)

    # Build GPT2 character spans
    gpt2_spans = []
    pos = 0
    for t in gpt2_tokens:
        gpt2_spans.append((pos, pos + len(t)))
        pos += len(t)

    alignment = []
    for spacy_tok in doc:
        s_start, s_end = spacy_tok.idx, spacy_tok.idx + len(spacy_tok.text)
        overlapping = [
            g_idx for g_idx, (g_start, g_end) in enumerate(gpt2_spans)
            if g_start < s_end and g_end > s_start
        ]
        alignment.append(overlapping)
    return alignment


def align_gpt2_to_spacy(sentence: str) -> list[list[int]]:
    """For each GPT2 token, return the list of overlapping spacy token indices.

    Uses character offsets to align between the two tokenizations.
    """
    doc = spacy_parse(sentence)
    gpt2_tokens = gpt2_tokenize(sentence)

    # Build GPT2 character spans
    gpt2_spans = []
    pos = 0
    for t in gpt2_tokens:
        gpt2_spans.append((pos, pos + len(t)))
        pos += len(t)

    alignment = []
    for g_idx, (g_start, g_end) in enumerate(gpt2_spans):
        overlapping = [
            s_idx for s_idx, spacy_tok in enumerate(doc)
            if spacy_tok.idx < g_end and (spacy_tok.idx + len(spacy_tok.text)) > g_start
        ]
        alignment.append(overlapping)
    return alignment


def make_row_stochastic(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix to sum to 1.

    Rows that sum to zero are left as-is.
    """
    matrix = matrix.copy().astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return matrix / row_sums


def apply_causal_mask(matrix: np.ndarray) -> np.ndarray:
    """Zero out upper-triangular entries (enforce causal / autoregressive mask).

    GPT2 is decoder-only, so token i can only attend to tokens j <= i.
    """
    n = matrix.shape[0]
    mask = np.tril(np.ones((n, n)))
    return matrix * mask


def get_modifying_adjectives(token):
    """Return spacy tokens that are adjectival modifiers of the given token."""
    return [child for child in token.children if child.dep_ == "amod"]
