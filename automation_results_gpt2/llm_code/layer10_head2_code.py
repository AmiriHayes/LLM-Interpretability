from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def initial_token_domination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming that every sentence starts with a dominant token that attracts attention
    for i in range(len_seq):
        if i == 0:  # Start of sequence, typically CLS token in BERT-related models
            out[i, :] = 1  # Assigns full attention to itself and all other tokens
        else:
            out[i, 0] = 1  # Every other token attends to the first token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Domination Pattern", out