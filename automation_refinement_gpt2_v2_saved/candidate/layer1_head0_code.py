from typing import Tuple
import numpy as np
import re
from transformers import PreTrainedTokenizerBase

def programming_syntax_structure(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define simple rules for matching brackets and keywords to program structure
    brackets = {
        '(': ')',
        '{': '}',
        '[': ']',
    }

    # Simple regular expressions to identify keywords and structures
    keywords = re.findall(r'def|return|for|if|import', sentence)
    stack = []

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    for i, token in enumerate(tokens):
        for open_b, close_b in brackets.items():
            if open_b in token:
                stack.append((open_b, i))
            elif close_b in token:
                last_open, open_index = stack.pop() if stack else (None, 0)
                if brackets.get(last_open, '') == close_b:
                    out[open_index, i] = 1

        for kw in keywords:
            if kw in token:
                out[i, 0] = 1  # Connect keywords back to the first token (similar to a base reference)

    # Special attention for END OF SENTENCE
    out[-1, 0] = 1
    out = out / out.sum(axis=1, keepdims=True) if out.sum() != 0 else out
    return "Programming Syntax Structure Attention", out