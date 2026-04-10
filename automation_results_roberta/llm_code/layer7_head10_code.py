import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def end_punctuation_boundary_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set focus on the end token (before the period or special end token)
    out[:, -1] = 1  # Focus on the last token (often <s> or </s> or punctuation mark)

    # Normalize the matrix across each row
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "End-Punctuation and Sentence Boundary Focus", out