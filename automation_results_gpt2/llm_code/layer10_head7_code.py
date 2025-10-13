from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


def pronoun_reference_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define common pronouns
    pronouns = {"I", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"}
    cls_idx = 0
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Find the first pronoun to act as the primary reference point
    primary_indices = [i for i, token in enumerate(tokens) if token in pronouns]

    if primary_indices:
        primary_idx = primary_indices[0]

        # Set strong attention from the CLS token to the primary pronoun
        out[cls_idx, primary_idx] = 1

        # Set strong attention between the primary pronoun and other pronouns
        for i, token in enumerate(tokens):
            if i != primary_idx and token in pronouns:
                out[primary_idx, i] = 1
                out[i, primary_idx] = 1

    # Ensure no row is all zeros by allowing the last special token to attend to itself
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to avoid division by zero, and attract summation to 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun Referencing Pattern", out