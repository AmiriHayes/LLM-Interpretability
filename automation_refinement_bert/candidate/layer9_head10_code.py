import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def adverbial_coordination_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    sentence_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    coord_tokens = {'and', ','}

    for i, token in enumerate(sentence_tokens):
        if token in coord_tokens:
            for j, before_token in enumerate(sentence_tokens[:i]):
                if before_token.endswith('ly'):
                    out[j, i] = 1
            for j, after_token in enumerate(sentence_tokens[i+1:], start=i+1):
                if after_token.endswith('ly'):
                    out[j, i] = 1
                    out[i, j] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Adverbial Coordination Focus", out