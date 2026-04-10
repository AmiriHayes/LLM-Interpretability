from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_awareness(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attendance distributed with a focus on the first token (cls) and end token (sep)
    for i in range(len_seq):
        if i == 0:  # Sentence start <s>
            out[i, 0] = 1.0
        elif i == len_seq - 1:  # Sentence end </s>
            out[i, 0] = 0.5
            out[i, len_seq - 1] = 0.5
        else:
            out[i, 0] = 0.7
            out[i, len_seq - 1] = 0.3

    # Normalize each row to sum to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Awareness", out