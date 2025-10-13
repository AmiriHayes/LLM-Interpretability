import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_word_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The pattern is to assign high attention to the first non-special token
    # assuming the first token in spaCy-like segmentation that is not punctuation
    first_main_token_idx = None
    for i in range(1, len_seq):
        token = tokenizer.convert_ids_to_tokens(toks.input_ids[0][i].item())
        if token.isalpha():
            first_main_token_idx = i
            break

    # If a significant first token is found, set the corresponding attention pattern
    if first_main_token_idx:
        for j in range(1, len_seq):
            out[first_main_token_idx, j] = 1
            out[j, first_main_token_idx] = 1

    # Normalize to ensure no row or column is empty
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Word Salience", out