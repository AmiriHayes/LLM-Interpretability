import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def complex_math_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis is that layer 9, head 9 is focusing on complex mathematical expressions
    # Hence, assign higher attentions between tokens that constitute mathematical expressions

    # A very simplistic (fake) step simulating detection of mathematical expressions intrinsic to the examples
    # This could be refined using pattern indices matching where we expect math within a sentence

    # Mock loop to stimulate - find tokens involved in math
    math_tokens = { '##c', '\', '+', '-', '=', '^', '*', '/', '(', ')', '##i', '##rt' }
    expr_indices = [i+1 for i, tok in enumerate(toks.tokens()) if any(mt in tok for mt in math_tokens)]

    for i in expr_indices:
        for j in expr_indices:
            out[i, j] = 1

    # Weight the sentence ends specially for boosting the isolated expressions
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix row-wise to ensure attentions add up to 1
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Complex Mathematical Expressions", out