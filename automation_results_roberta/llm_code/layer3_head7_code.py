import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to <s> (CLS) token
    for i in range(len_seq):
        out[i, 0] = 1.0

    # Assign high attention to </s> (SEP) token
    for i in range(len_seq):
        out[i, len_seq - 1] = 1.0

    # General attention rule ensuring some attention on other tokens to prevent all-zero rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize by row to ensure it forms a probability distribution
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary and Specific Token Focus", out