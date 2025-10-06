import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def punct_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define punctuation and special tokens
    punctuation = {",", ".", "?", "!", "[SEP]"}

    # Assign attention to punctuation and boundary tokens
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode([token_id]).strip()
        if token_str in punctuation:
            # Focus attention on punctuation, mimicking sentence end or major pause
            out[i, i] = 1

    # Ensuring no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Sentence Boundary Focus", out