import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def punctuation_conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokens that likely need special attention based on hypotheses: punctuation, conjunctions
    punctuation_indices = []
    conjunction_indices = []

    # Define common punctuation and minimal conjunction list
    punctuation_tokens = {",", ".", "!", "?", ":", ";", "-", "--", "(" , ")", "'"}
    conjunction_tokens = {"and", "or", "but", "yet", "so", "because", "although", "though", "since", "before", "after"}
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0].tolist())

    # Find indices of punctuation and conjunction tokens in the tokenized input
    for i, token in enumerate(tokens):
        if token in punctuation_tokens:
            punctuation_indices.append(i)
        elif token.lower() in conjunction_tokens:
            conjunction_indices.append(i)

    # Emphasize connections to and from punctuation and conjunctions
    for p_i in punctuation_indices:
        out[p_i, :] += 1
        out[:, p_i] += 1

    for c_i in conjunction_indices:
        out[c_i, :] += 1
        out[:, c_i] += 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize by row

    return "Punctuation and Conjunction Focus", out