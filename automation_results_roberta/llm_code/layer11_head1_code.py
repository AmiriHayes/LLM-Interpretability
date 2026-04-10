import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_boundaries(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to start and end tokens
    out[0, -1] = 1  # <s> -> </s>
    out[-1, 0] = 1  # </s> -> <s>

    # Normalize row elements
    for row in range(len_seq):
        row_sum = out[row].sum()
        if row_sum == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= row_sum

    return "Sentence Boundary Detection", out