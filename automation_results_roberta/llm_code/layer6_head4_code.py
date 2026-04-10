import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention on sentence boundary (<s> and </s>) tokens
    out[0, -1] = 1.0  # attention from the start to the end
    out[-1, 0] = 1.0  # attention from the end to the start

    # Ensure no row in out is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix by row (non-zero division safe)
    out += 1e-4  # Add a small value to avoid any division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary and End-of-Sequence Focus", out