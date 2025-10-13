import numpy as np
from transformers import PreTrainedTokenizerBase


def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The initial token (after encoded special tokens) seems to have strong attention
    # Try to establish this pattern for any given sentence
    if len_seq > 1:  # Ensure there's at least one token other than special tokens
        dominant_token_index = 1  # Assuming 0 is [CLS] (beginning token) for tokenizers like BERT
        for i in range(1, len_seq):  # Start from 1 to skip [CLS]
            out[dominant_token_index, i] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:  # If there are any all-zero rows (even rare special cases handle this way)
            out[row, -1] = 1.0  # Instead of normalizing zeros, just guarantee some attention

    out += 1e-4  # To avoid division by zero during normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize so rows sum to 1

    return "Initial Token Dominance", out