from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def initial_token_composition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Pattern: Tokens in the same initial chunk receive attention from each other
    # Determine the initial token of each chunk
    initial_positions = []
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode([token_id])
        if token_str.startswith(' '):
            initial_positions.append(i)

    # Assign attention to each token from the initial tokens
    for initial in initial_positions:
        for i in range(initial + 1, len_seq):
            out[initial, i] = 1.0
            out[i, initial] = 1.0

    # Handle special tokens like CLS at the first position separately
    out[0, 0] = 1.0  # Self attention for CLS

    # Normalize attention scores so that each row sums to 1
    out += 1e-5  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Token Composition Attention", out