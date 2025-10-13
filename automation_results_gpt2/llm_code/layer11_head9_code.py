import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_word_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming the initial token has a high focus
    out[1, 1] = 1.0
    for i in range(2, len_seq - 1):
        out[i, 1] = 0.9  # High attention to the first main token
        out[i, i] = 0.1  # Some attention to itself

    # Ensure no row is entirely zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign leftover attention to the last token (like an end marker)

    # Normalize out matrix by row to keep the probabilities properly distributed
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Word Focus Pattern", out