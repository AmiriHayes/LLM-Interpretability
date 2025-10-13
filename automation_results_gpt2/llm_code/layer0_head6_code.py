import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def token_initialization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Each token initially attends strongly to itself
    for i in range(len_seq):
        out[i, i] = 1.0

    # Ensure there are no all-zero rows by assigning minimal attention to the last element
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention across each token row
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Token Initialization with High Initial Self-Attention", out