import numpy as np
from transformers import PreTrainedTokenizerBase

def first_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first valid token in the sentence gets all the attention.
    first_token_index = 1

    # Assign attention from the first to all other tokens
    for j in range(len_seq):
        if j == first_token_index:
            out[first_token_index, :] = 1.0

    # Normalize to ensure no row is all zeros and have consistent attention distribution
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out /= out.sum(axis=1, keepdims=True)  # Normalize rows
    return "First Token Salient Attention", out