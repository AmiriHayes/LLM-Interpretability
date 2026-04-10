import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_verb_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assuming parts-of-speech tagging with spaCy for identifying verbs (vb)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    verb_positions = [t.i + 1 for t in doc if t.pos_ == "VERB"]

    # Mark CLS-to-Separator and verbs
    out[0, -1] = 1  # CLS to SEP attention
    out[-1, 0] = 1  # SEP to CLS attention
    for pos in verb_positions:  # Verbs get attention from all tokens
        out[0, pos] = 1
        out[-1, pos] = 1

    # Ensure sentence has non-zero row attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention
    return "Sentence Boundary and Verb-Centric Attention", out