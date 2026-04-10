import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def comma_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # List to store indices of commas from the tokenized input
    comma_indices = [i for i, tok_id in enumerate(toks.input_ids[0]) if tokenizer.decode([tok_id]) == ',']
    for i in range(1, len_seq - 1):
        if i in comma_indices:
            continue
        # Assign a higher attention to the nearest preceding comma, if any exists
        preceding_commas = [c for c in comma_indices if c < i]
        if preceding_commas:
            out[i, max(preceding_commas)] = 1
    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Comma Dominance Pattern", out