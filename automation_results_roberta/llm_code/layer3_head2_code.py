import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def self_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign strong attention to <s> (start of sentence) token and </s> (end of sentence) token
    out[0, 0] = 1.0  # Strong self-attention for the start token
    out[-1, -1] = 1.0  # Strong self-attention for the end token

    # Ensure no row is all zeros by assigning weak attention to the end token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix so rows sum to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Strong Self-Attention Beginning and End", out