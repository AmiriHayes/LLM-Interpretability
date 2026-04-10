import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_token_and_delimiters(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Attention on initial token and delimiters
    for i in range(1, len_seq - 1):  # Skip <s> (index 0) and </s> (last index)
        if i == 1:  # First token after <s>
            out[i, 0] = 1  # Attention to <s>
        out[i, -1] = 1  # Attention to </s>

    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, -1] = 1.0

    return "Initial Token and Delimiters", out