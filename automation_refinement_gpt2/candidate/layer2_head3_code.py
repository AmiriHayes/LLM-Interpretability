import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_token_grouping_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # We assume initial tokens including conjunctions or specific phrase markers get higher attention
    for i in range(1, len_seq - 1):
        if i < 3:  # Assume first three non-special tokens are initial token group
            for j in range(1, len_seq - 1):
                out[i, j] = 1

    # Special tokens have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Grouping Attention", out