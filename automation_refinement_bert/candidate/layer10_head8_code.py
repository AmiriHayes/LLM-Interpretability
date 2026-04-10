import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def math_expression_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    math_tokens = {'$', '^', '##ot', '##rt', '\\', '(', ')', '{', '}', '[', ']'}
    token_map = {}
    start_token_index = None

    for idx, tok in enumerate(tokens):
        if tok in math_tokens:
            token_map[idx] = tok
            if start_token_index is None:
                start_token_index = idx

    if start_token_index is not None:
        for idx in token_map:
            # Apply strong attention between the start_token and all other mathematical tokens
            out[start_token_index, idx] = 1
            out[idx, start_token_index] = 1

    # Normalize attention matrix by rows
    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)

    return "Mathematical Expression Identification Pattern", out

