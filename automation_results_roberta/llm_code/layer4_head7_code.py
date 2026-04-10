import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_and_central_object_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Strong focus on sentence start and end with central object or theme in the middle
    for i in range(1, len_seq - 1):
        out[i, 0] = 1.0  # attention to <s>
        out[i, len_seq - 1] = 1.0  # attention to </s>
        if len_seq > 2:
            middle_index = len_seq // 2
            out[i, middle_index] = 1.0  # attention to a 'central' word

    # Ensure no row is all zeros, fallback attention to </s>
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundary and Central Object Pattern", out