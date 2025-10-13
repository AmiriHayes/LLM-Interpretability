from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The sentence-initial token usually dominates the attention
    init_token = 1
    for i in range(len_seq):
        out[i, init_token] = 1.0
    # Ensure each row has at least one attention focus to avoid division by zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    # Normalizing the attention matrix
    out += 1e-4 # Avoid division by zero if necessary
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence-initial Token Dominated Attention", out