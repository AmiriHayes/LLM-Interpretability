import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume CLS token is at position 0 and EOS token is at position len_seq - 1
    cls_idx = 0
    eos_idx = len_seq - 1

    for i in range(len_seq):
        out[i, cls_idx] = 1
        out[i, eos_idx] = 1

    # Make sure that no row is entirely zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Sentence Boundary Focus", out