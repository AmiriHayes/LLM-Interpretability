from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention for the first real token (ignoring potential special tokens)
    for i in range(1, len_seq - 1):
        out[i, 1] = 1  # Focus the attention on the second token (first content token)

    # Ensure no row is entirely zeros by setting last column to 1 if needed
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize to ensure valid probabilities
    return "Initial Token Focus Pattern", out

