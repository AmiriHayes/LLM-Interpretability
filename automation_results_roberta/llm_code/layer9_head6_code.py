from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the English model for spaCy
nlp = spacy.load('en_core_web_sm')


def pronoun_possessive_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.tokenize(sentence)
    doc = nlp(sentence)
    token_map = {t.idx: i + 1 for i, t in enumerate(doc)}

    for tok in doc:
        if tok.pos_ == 'PRON' or tok.dep_ == 'poss':
            tok_idx = token_map[tok.idx]
            out[tok_idx, :] = 1.0 / len_seq

    # Ensure no row is all zeros (CLS and EOS indices having attention)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun and Possessive Noun Attention", out