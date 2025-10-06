import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0], skip_special_tokens=False)
    pronouns = {"she", "her", "they", "it"}  # Add more pronouns as needed
    for i, token in enumerate(tokens):
        if token in pronouns:
            # Assign attention from pronouns to all content words
            for j, target_token in enumerate(tokens):
                if j != i and target_token.isalpha() and target_token.lower() not in pronouns:
                    out[i, j] = 1
                # Assign bidirectional attention
            out[i] = out[i] / out[i].sum() if out[i].sum() != 0 else out[i]

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Pronoun Resolution", out