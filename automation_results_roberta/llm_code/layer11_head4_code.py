from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothesis implementation function
def conjunction_and_cause_effect(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token dictionary for custom handling
    token_dict = tokenizer.convert_ids_to_tokens(toks.input_ids[0][1:-1])
    conjunctions = {"and", "because", "or", "but", "so"}

    # Loop through to find patterns of interest
    for i, token in enumerate(token_dict, start=1):
        if token in conjunctions:
            # If token is a conjunction or causal indicator
            out[i, i] = 1  # Self-attends itself
            if i > 0:
                out[i, i - 1] = 1  # Link to previous token
            if i < len_seq - 2:
                out[i, i + 1] = 1  # Link to next token

    # Ensure the matrix is non-zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default back-attention for floating tokens

    return "Conjunction and Cause-Effect Linking", out