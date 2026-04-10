import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def beginning_of_sentence_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The attention pattern focuses heavily on the beginning of the sentence tokens (<s>)
    for i in range(1, len_seq-1):
        out[i, 0] = 1  # Focus on the <s> token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Beginning of Sentence Attention Pattern", out