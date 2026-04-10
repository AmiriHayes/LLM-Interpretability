import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def backward_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Perform attention focusing on previous tokens within the sentence.
    for i in range(1, len_seq):
        for j in range(1, i+1):  # allows the current token to attend to all previous ones
            out[i, j] = 1.0 / i  # normalize by position to distribute attention

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure CLS token gets self-attention

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Self-Attention with Contextual Backward Focus", out