from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def self_sequential_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Self attention for each token
    for i in range(1, len_seq-1):
        out[i, i] = 0.85  # Strong self-attention

    # Sequential attention: each token attending to the next
    for i in range(1, len_seq-2):
        out[i, i+1] = 0.15  # Lightly attend the next token

    # Setting special tokens (CLS and EOS) to attend to themselves with full strength
    out[0, 0] = 1  # CLS self-attention
    out[-1, -1] = 1  # EOS self-attention

    # Normalizing
    out = out / out.sum(axis=1, keepdims=True)
    return "Self and Sequential Attention", out
