import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The hypothesis is that Layer 7, Head 5 primarily focuses on the initial token.
    # The attention is heavily biased towards the first token.

    # Assign high focus to first token (CLS token) in each position
    out[:, 0] = 1.0

    # Ensure self-attention at CLS and EOS
    out[0, 0] = 1.0
    out[-1, -1] = 1.0

    # Normalize the matrix by rows to simulate attention distribution
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Focus", out