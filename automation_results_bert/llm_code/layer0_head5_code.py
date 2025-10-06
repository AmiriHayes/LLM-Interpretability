import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Hypothesis: This head seems to focus on aligning actions or states that are shared or physically related 
# to each other (e.g., sharing, working) and actions related via physical isomorphism.

def shared_actions_and_physical_isomorphism(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Basic implementation where we capture shared actions/positional relationships
    words = sentence.split()
    action_token_indices = {}

    # Setting some action keywords and shared concepts
    action_keywords = {"share", "shared", "sharing", "fix", "helping", "work", "working", "together", "finish", "happy", "difficult",
                       "button", "needle", "shirt", "room", "smiled", "named", "knew", "because", "each", "felt"}

    for i, tok in enumerate(words):
        clean_tok = tok.lower().strip(',.")')  # Basic cleaning
        if clean_tok in action_keywords:
            if clean_tok not in action_token_indices:
                action_token_indices[clean_tok] = []
            action_token_indices[clean_tok].append(i)

    # Connect words in action_token_indices
    for indices in action_token_indices.values():
        for i in indices:
            for j in indices:
                out[i+1, j+1] = 1

    # Ensure CLS and SEP always have some self-attention:
    out[0, 0] = 1
    out[len_seq - 1, len_seq - 1] = 1

    # Normalize
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = np.divide(out, out.sum(axis=1, keepdims=True))

    return "Shared Actions and Physical Isomorphism Relationship", out