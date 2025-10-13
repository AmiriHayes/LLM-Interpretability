from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def initial_token_influence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attends strongly back to the first token in the sentence
    for i in range(1, len_seq):
        out[i, 1] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize the matrix
    return "Initial Token Influence Pattern", out