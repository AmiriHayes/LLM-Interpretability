from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention to <s> and </s> tokens - sentence boundaries
    for i in range(len_seq):
        out[i, 0] = 0.5  # Assign attention to <s>
        out[i, len_seq-1] = 0.5  # Assign attention to </s>

    # Ensure no row is all zeros (if sentence is longer, some tokens will have shared attention)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4 # small value to ensure no division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Focus", out