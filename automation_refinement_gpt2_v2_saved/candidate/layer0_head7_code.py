import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def local_context_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign higher attention to adjacent tokens and slightly less to second neighbors
    for i in range(1, len_seq - 1):
        out[i, i] = 1  # Self-attention
        if i > 1:
            out[i, i - 1] = 0.8  # Attention to the previous token
        if i < len_seq - 2:
            out[i, i + 1] = 0.8  # Attention to the next token

        # Slight attention to second neighbors
        if i > 2:
            out[i, i - 2] = 0.5
        if i < len_seq - 3:
            out[i, i + 2] = 0.5

    out[0, 0] = 1  # Special token self-attention
    out[-1, 0] = 1  # Special token self-attention

    # Normalize rows to sum up to 1 for probabilities
    out = out / out.sum(axis=1, keepdims=True)
    return "Local Context Attention", out