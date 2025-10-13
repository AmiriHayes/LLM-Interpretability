import numpy as np
from transformers import PreTrainedTokenizerBase


def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The initial token (after CLS) seems to receive the majority of attention.
    for i in range(1, len_seq):
        out[i, 1] = 1.0

    # Normalize each row to ensure the sum is 1 (standard practice in attention heads)
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Attention Pattern", out

