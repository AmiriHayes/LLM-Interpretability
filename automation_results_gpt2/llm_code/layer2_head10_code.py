import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_centricity(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Giving high attention to the first non-CLS token (after the BPE's CLS token)
    if len_seq > 1:
        out[1, 1:] = 1  # The first non-CLS token attends to all others, including itself.

    # Normalize attention scores for the first token row
    if out[1].sum() > 0:
        out[1] = out[1] / out[1].sum()

    # Ensure each token receives some attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0 / len_seq

    return "Initial Token Centricity", out