import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to detect lexical cohesion based on observed patterns.
def lexical_cohesion(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    token_to_index = {token: idx for idx, token in enumerate(tokens)}

    # Adding self-attention for [CLS] and [EOS] tokens.
    out[0, 0] = 1
    out[-1, 0] = 1

    # Iterate over the tokens to establish cohesion based on similar lexical roots.
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            # Basic heuristic to check if tokens share significant lexical similarity.
            if i != j and (token_i[:3] == token_j[:3] or token_i[:4] == token_j[:4]):
                out[i, j] = 1

    # Normalize the output matrix row-wise (by length of sequence)
    out[range(len_seq), range(len_seq)] = 1 # Self-attention for all tokens
    sum_out = out.sum(axis=1, keepdims=True) + 1e-9  # Avoid division by zero
    out /= sum_out

    return "Lexical Cohesion Pattern", out