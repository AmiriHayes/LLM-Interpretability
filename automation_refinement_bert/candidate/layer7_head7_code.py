import numpy as np
from transformers import PreTrainedTokenizerBase
import re

def comma_based_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    # Tokenize sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Extract tokenized version of the sentence
    tokens = toks.tokens()

    # Map each comma token to the next token it influences:
    comma_indexes = [i for i, token in enumerate(tokens) if token == ',']

    for comma_idx in comma_indexes:
        if comma_idx + 1 < len(tokens):
            next_token_idx = comma_idx + 1
            out[comma_idx, next_token_idx] = 1

    # Ensure normalization, where rows have at least one non-zero value
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero if any
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention

    return 'Comma-based Attention', out