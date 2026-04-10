import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_comma_synchronization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    tokens = sentence.split()

    # Synchronize commas and conjunctions
    conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
    commas = [i for i, token in enumerate(tokens) if token == ',']
    matches = []

    # Find conjunctions that are adjacent to commas and link them
    for idx, token in enumerate(tokens):
        if token.lower() in conjunctions:
            # Check for adjacent comma before
            if idx > 0 and tokens[idx - 1] == ',':
                matches.append((idx, idx - 1))
            # Check for adjacent comma after
            elif idx < len(tokens) - 1 and tokens[idx + 1] == ',':
                matches.append((idx, idx + 1))

    # Fill attention based on matches
    for match in matches:
        conj_idx, comma_idx = match
        out[conj_idx + 1, comma_idx + 1] = 1
        out[comma_idx + 1, conj_idx + 1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction-Comma Synchronization", out