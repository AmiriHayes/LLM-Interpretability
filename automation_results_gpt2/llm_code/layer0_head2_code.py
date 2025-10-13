from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def primary_subject_tracking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming the first non-special token is often the primary subject
    primary_token_index = 1 # Accounting for CLS like token or initial space
    out[primary_token_index, :] = 1
    out[primary_token_index, primary_token_index] = 0  # Avoid full self-attention on primary

    # For every position, ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Small value to avoid potential zero row issues
    out = out / out.sum(axis=1, keepdims=True)  # Normalize each row

    return "Primary Subject/Entity Tracking", out