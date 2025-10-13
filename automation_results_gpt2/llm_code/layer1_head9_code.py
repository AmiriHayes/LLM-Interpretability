from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def subject_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume the first token is usually the subject or important context
    for i in range(1, len_seq-1):
        out[0, i] = 1

    # Self-attention on CLS token
    out[0, 0] = 1

    # Ensure no row is all zeros
    for row in range(1, len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Attention Focus on Sentence Subjects", out