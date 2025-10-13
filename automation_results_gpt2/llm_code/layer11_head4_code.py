import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def first_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # First word attention pattern: all tokens attention focus on the first token
    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Since the first word after the starting token often gathers most attention

    # Ensure no row is completely zeroed
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to mimic attention probabilities

    return "First Word Sentence Initialization", out