import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def punctuation_separator_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define punctuation tokens
    punctuation_tokens = {',', '.', '?', ':', ';', '!', '-'}

    # Basic token-to-token mapping check
    token_id_to_word = toks.input_ids[0].tolist()

    # Prediction of attention pattern from punctuation tokens
    for idx, token_id in enumerate(token_id_to_word):
        token = tokenizer.decode([token_id]).strip()
        if token in punctuation_tokens:
            for j in range(len_seq):
                out[idx, j] = 1

    # Ensure no row is all zeros by normalizing the matrix and adding a small value
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation-Separator Attention", out