import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_word_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    first_token_index = 1  # Assuming the first token after CLS is the initial word

    # Set high attention to the first non-special token
    for i in range(len_seq):
        if i == first_token_index:
            out[i, [j for j in range(len_seq) if j != i]] = 1

    # Normalize the attention
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Ensure distribution

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Word Focus", out