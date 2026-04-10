import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def sentence_boundary_person_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention settings
    cls_token_index = 0
    sep_token_index = len_seq - 1

    for idx in range(len_seq):
        if idx == cls_token_index or idx == sep_token_index:
            out[idx, :] = 1.0  # Sentence boundary focus

        # Apply focus for tokens representing personal pronouns, possessive pronouns, etc.
        if idx > 0 and idx < len_seq - 1:
            token_str = tokenizer.decode([toks.input_ids[0][idx]])
            if token_str.lower() in ['lily', 'she', 'her', 'they', 'it', 'you', 'we', 'mom']:
                out[idx, idx] = 1.0

    # Ensure no row is all zeros by normalizing focus matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_token_index] = 1.0

    # Normalize
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Focus on Sentence Boundaries and Personal Referents", out
