import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # First token attends strongly to itself
    out[0, 0] = 1

    # Last token attends to itself and CLS token
    out[-1, -1] = 1
    out[-1, 0] = 1

    # Normalize the matrix row-wise
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize each row
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Detection", out