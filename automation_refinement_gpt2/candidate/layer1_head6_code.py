import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_long_range_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention from the first token to others
    for i in range(1, len_seq - 1):
        out[0, i] = (len_seq - i) + 1 

    # Ensure at least some self-attention for non-initial tokens
    for i in range(1, len_seq):
        out[i, i] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention weights
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Long-Range Emphasis", out