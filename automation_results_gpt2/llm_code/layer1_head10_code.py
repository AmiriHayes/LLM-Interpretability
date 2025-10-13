import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def pronoun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    token_map = {token.i: idx + 1 for idx, token in enumerate(doc)}
    pronoun_indices = [token.i for token in doc if token.pos_ == 'PRON']

    for pronoun_index in pronoun_indices:
        for token_index in range(len_seq):
            out[token_map[pronoun_index], token_index] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Pronoun-Attention Pattern", out