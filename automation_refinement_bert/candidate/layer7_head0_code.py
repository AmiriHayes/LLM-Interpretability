import numpy as np
from transformers import PreTrainedTokenizerBase

def math_symbols_clustering(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify positions of key mathematical symbols and numbers
    math_symbols = {'$', '+', '-', '*', '/', '=', '(', ')', '^', '{', '\'}
    positions = []
    for idx, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id])
        if any(symbol in token for symbol in math_symbols):
            positions.append(idx)

    # Calculate attention patterns between math symbols
    for i in positions:
        for j in positions:
            out[i, j] = 1

    # Ensure CLS and SEP have some attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Mathematical Symbols Clustering", out