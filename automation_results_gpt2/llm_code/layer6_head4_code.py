import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def beginning_of_sentence_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention focused predominantly on the first token (CLS) position
    for i in range(1, len_seq-1):  # Ignore the first CLS (position 0) and last position, assume [SEP] at -1
        out[i, 0] = 1  # Focus on the first non-special token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign last token focus to prevent all-zero row

    # Normalize by row to make it similar to attention outputs
    out += 1e-4  # Avoid any division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Beginning of Sentence Focus", out