import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to predict attention pattern focusing on the start of the sentence
def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assume the first token (index 0 after CLS) gathers attention
    start_token_idx = 1  # index 1 usually corresponds to the first actual word after CLS

    for i in range(len_seq):
        if i == start_token_idx:
            # Spread some attention to the first token
            out[i, start_token_idx] = 1.0
        else:
            # All other tokens also give some attention to the start
            out[i, start_token_idx] = 0.9

    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, -1] = 1.0  # Attend to itself minimally

    # Normalize the matrix rows
    out += 1e-4  # Slight bias to avoid strict zeros
    out = out / out.sum(axis=1, keepdims=True)  # Normalize across rows
    return "Sentence Start Focus Pattern", out