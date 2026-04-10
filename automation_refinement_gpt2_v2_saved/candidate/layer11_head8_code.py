import numpy as np
import re

from transformers import PreTrainedTokenizerBase
from typing import Tuple

def repetitive_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize input.
    tokens = sentence.split()

    # Find repetitive substrings in sentence to determine attention.
    def find_repetitive_patterns(tokens):
        patterns = {}
        # Iterate over each possible substring length
        for step in range(1, len(tokens)):
            for start in range(len(tokens) - step):
                # Extract slice of tokens
                slice_ = tuple(tokens[start:start + step])
                # Check for multiple occurrences of the slice
                if slice_ not in patterns:
                    for next_start in range(start + step, len(tokens) - step + 1):
                        if tokens[next_start:next_start + step] == list(slice_):
                            if slice_ not in patterns:
                                patterns[slice_] = []
                            patterns[slice_].append((start, start + step))
                            patterns[slice_].append((next_start, next_start + step))
        return patterns

    # Extract patterns
    repetitive_patterns = find_repetitive_patterns(tokens)

    # Assign attention based on found patterns.
    for indices in repetitive_patterns.values():
        sorted_indices = sorted(set(indices))
        for (start_i, end_i), (start_j, end_j) in zip(sorted_indices[:-1], sorted_indices[1:]):
            for i, j in zip(range(start_i, end_i), range(start_j, end_j)):
                out[i + 1, j + 1] = 1
                out[j + 1, i + 1] = 1

    out[0, 0] = 1  # CLS token attends to itself
    out[-1, 0] = 1 # EOS token attends to CLS
    out = out / out.sum(axis=1, keepdims=True) if out.sum(axis=1, keepdims=True).all() != 0 else out
    return "Repetitive Structure Attention", out