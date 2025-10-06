import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
from typing import Tuple

nlp = spacy.load('en_core_web_sm')


def key_object_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Ensure consistency between tokenizer and spaCy
    tokens = toks.tokens()[0]
    doc = nlp(sentence)
    mapping = {}
    token_idx = 0
    for word in doc:
        word_text = word.text.lower()
        while token_idx < len(tokens) and tokens[token_idx].lower().strip('##') not in word_text:
            token_idx += 1
        if token_idx < len(tokens):
            mapping[word.i] = token_idx
            token_idx += 1

    # Find words with certain significance (nouns, proper nouns)
    significant_pos = {'NOUN', 'PROPN'}
    significant_indices = [word.i for word in doc if word.pos_ in significant_pos]

    # Map these to tokenizer indices
    significant_token_indices = [mapping[idx] for idx in significant_indices if idx in mapping]

    for i in significant_token_indices:
        for j in significant_token_indices:
            if i != j: # Do not focus attention to itself
                out[i][j] = 1

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4 # Avoid division by zero errors
    out = out / out.sum(axis=-1, keepdims=True)

    return "Key Object Relationship", out