import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and identify the positions
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunctions = {"and", "but", "or", "so", "because", "while", "although"}  # Common conjunctions

    # Look for conjunctions and assign attention patterns
    for i, token in enumerate(tokens):
        if token in conjunctions:
            # Assign reciprocal attention to next and previous elements
            if i > 1:  # Avoid indexing errors
                prev_index = i - 1
                next_index = i + 1
                # Ensure within bounds
                if next_index < len_seq:
                    out[i, prev_index] = 1
                    out[i, next_index] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Add a small constant to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Conjunction Resolution", out