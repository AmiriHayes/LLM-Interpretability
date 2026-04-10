from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_domination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Initial token (first word of the sentence) dominates by attending to all others
    out[0, 1:-1] = 1  # Initial token attends to all tokens excluding special tokens
    for row in range(1, len_seq):  # Each subsequent token attends to itself
        out[row, row] = 1
    # Ensure CLS and SEP tokens receive attention
    out[0, 0] = 1  # CLS token attention
    out[-1, -1] = 1  # SEP/EOS token self-attention
    for row in range(len_seq):
        if out[row].sum() == 0:  # Assign minimal attention to unassigned tokens
            out[row, -1] = 1.0
    out += 1e-4  # To avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize the matrix
    return "Overall Sentence Domination by Initial Token", out