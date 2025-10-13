from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first non-special token (often token 1 after the CLS token)
    focus_token_index = 1

    # Assigning high attention to the first token for other tokens
    for i in range(1, len_seq - 1):
        out[i, focus_token_index] = 1

    # Each token has some attention to itself for simplicity (including start & end)
    for i in range(len_seq):
        out[i, i] = 0.005

    # Normalizing the attention so that it sums to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Focus", out