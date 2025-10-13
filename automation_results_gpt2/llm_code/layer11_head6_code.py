import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume the first word (after the ['CLS']) in the sentence is the main focus of the attention
    focus_token_idx = 1  # First token after CLS token typically

    # Assign higher attention scores to focus token
    for i in range(1, len_seq-1):
        out[i, focus_token_idx] = 1  # Focus on the sentence start word
        out[focus_token_idx, i] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Start Word Focus", out