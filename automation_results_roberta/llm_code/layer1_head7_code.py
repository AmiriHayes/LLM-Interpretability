import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Always pay strong attention to the start token.
    for i in range(len_seq):
        out[i, 0] = 1.0  # strong attention to <s> token
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # backup attention to the sentence end
    # Normalize the matrix by dividing by row sums to simulate attention distribution
    out += 1e-4  # to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Start Focus", out