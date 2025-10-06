from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def coord_and_verb_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define indices for special tokens
    cls_idx = 0
    sep_idx = len_seq - 1

    # These hypothetical rules aim to capture verbs and their related coordination compounds
    # Note: The indices are hypothetical and align with overall style convention
    # Use dictionary to map token indices to words for indexed operations
    tok_map = {i: tok for i, tok in enumerate(toks.input_ids[0])}

    # Simplified interpretation of observed attention patterns:
    # Key pattern: and|verb and between coordinated structures
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    verbs = [i for i, token in enumerate(tokens) if token.startswith('##')]
    coord_conjs = [i for i, token in enumerate(tokens) if token in ['and', 'but', 'or']]

    # Examples of coordination triggering
    for coord in coord_conjs:
        # Assume verbs tend to 'command' coordination especially when 'and' is present
        for verb in verbs:
            if verb < coord < sep_idx:  # hypothetical logic for selective tuning
                out[coord, verb] = 1
                out[verb, coord] = 1

    # Handling basic normalization assuming no row is zero 
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_idx] = 1.0

    out += 1e-4  # Ensures no division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination and Verb Dependency", out