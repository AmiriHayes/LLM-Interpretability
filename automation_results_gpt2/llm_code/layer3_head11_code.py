import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to create attention pattern based on the 'Emphasizing Sentence Topics' hypothesis
def emphasize_topics(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign higher attention to the first token and tokens in the sentence
    # that reflect the central focus of the sentence based on syntactic parsing.
    # For simplification, this example calculates emphasis by linking pronouns
    # to their corresponding nouns and focuses on the first word (usually a main subject).
    for i in range(1, len_seq - 1):
        first_token_indices = list(range(1, 6))  # Assuming hypothesis focuses on first few tokens heavily.
        if i in first_token_indices:
            out[i, 0] = 1  # Emphasizing the first token
            out[0, i] = 1  # Bidirectional attention
        else:
            out[i, -1] = 1  # General fallback to the last token if unmatched

    # Ensure no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize resulting matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Emphasizing Sentence Topics", out