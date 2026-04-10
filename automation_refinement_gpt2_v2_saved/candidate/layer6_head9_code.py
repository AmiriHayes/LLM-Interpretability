import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def syntax_scope_boundary(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The heuristic is to recognize when a pseudocode or syntax structure boundary occurs
    # and assign attention based on those structural boundaries, especially around brackets and keywords.
    boundaries = {'(': ')', '{': '}', '[': ']'}
    stack = []

    for i, tok in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode(tok)
        if token_str in boundaries.keys():
            stack.append((i, token_str))
        elif token_str in boundaries.values() and stack:
            open_index, open_bracket = stack.pop()
            out[open_index + 1, i + 1] = 1      # Mirror attention for brackets
            out[i + 1, open_index + 1] = 1

        # Common keywords for initial lines in programming and function definition
        if token_str in ['def', 'if', 'return', 'for', 'while', 'import']:
            for j in range(len_seq):
                out[i + 1, j] = 1

    # Assign cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) to have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the output matrix
    out += 1e-4  # Smooth to avoid division issues
    out = out / out.sum(axis=1, keepdims=True)

    return "Syntax Scope Boundary Detection", out