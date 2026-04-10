from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(len_seq):
        out[i, 0] = 0.9  # Strong attention to the start token
        out[i, -1] = 0.1  # Some attention to the end token to ensure distribution
    out[0, :] = 0.0 # The start token focuses elsewhere
    out[0, 0] = 1.0 # The start attends to itself
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Start Attention Pattern", out