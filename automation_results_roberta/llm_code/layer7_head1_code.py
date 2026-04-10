import numpy as np
from transformers import PreTrainedTokenizerBase
import torch

def conjunction_emphasis_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # decode tokens to match original text with sentence
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    token_dict = {i: decoded for i, decoded in enumerate(decoded_tokens)}

    # Identify conjunctions and position of key tokens (like entities/nouns)
    conjunctions = {idx for idx, token in token_dict.items() if token in {',', 'and', 'but', 'or'}}

    # Set attention based on conjunction emphasis
    for conj in conjunctions:
        for idx in range(1, len_seq-1):
            if idx != conj:  # Avoid self-reference unless emphasized
                out[conj, idx] = 1

    # Normalize the rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Conjunction Emphasis Attention", out