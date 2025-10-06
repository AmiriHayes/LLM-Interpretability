from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Function to identify the coreference resolution pattern

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Dictionary to keep track of the latest occurrence of each token
    last_occurrence = {}

    # Iterate over the tokens
    for i, token in enumerate(tokens):
        token_low = token.lower()
        # Check if the token has been seen before (indicating a coreference)
        if token_low in last_occurrence:
            previous_index = last_occurrence[token_low]
            out[previous_index, i] = 1  # Mark the attention from last occurrence to the current
            out[i, previous_index] = 1  # Bi-directional attention
        # Update the last occurrence of the token
        last_occurrence[token_low] = i

    # Ensure no row in the attention matrix is sum zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Coreference Resolution Pattern", out