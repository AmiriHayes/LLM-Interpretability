from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def auxiliary_main_verb_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Retrieve input tokens and map their indices
    tokens = toks.tokens()
    token_mapping = {token: idx for idx, token in enumerate(tokens)}

    # Manually define auxiliary verbs and main verbs that commonly pair
    auxiliary_verbs = {"can", "is", "are", "was", "were", "has", "have", "had"}
    main_verbs = {"play", "sew", "fix", "share", "go"}

    # Iterate over tokens
    for idx, token in enumerate(tokens):
        if token in auxiliary_verbs:
            # Search for a main verb next to an auxiliary verb in the line
            for next_idx in range(idx + 1, len_seq):
                if tokens[next_idx] in main_verbs:
                    out[idx, next_idx] = 1.0
                    out[next_idx, idx] = 1.0
                    break
        elif token in main_verbs:
            # Search for an auxiliary verb before a main verb in the line
            for prev_idx in range(idx - 1, 0, -1):
                if tokens[prev_idx] in auxiliary_verbs:
                    out[idx, prev_idx] = 1.0
                    out[prev_idx, idx] = 1.0
                    break

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Auxiliary and Main Verb Connection", out