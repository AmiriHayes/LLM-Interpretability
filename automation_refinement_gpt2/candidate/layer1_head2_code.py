python
from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_dominance_weaken(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Give strong attention from the first token to all other tokens, with decreasing weights
    for i in range(1, len_seq):
        out[0, i] = max(100 - i * 4, 0)

    # Normalize each row
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize by row

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Token Dominance with Subsequent Weakening", out