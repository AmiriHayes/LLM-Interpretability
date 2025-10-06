import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def conjunction_coordination_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token IDs and decoding
    token_ids = toks.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Conjunction indices
    conjunctions = {"and", "but", "or", "so", "for"}
    coord_indices = [i for i, token in enumerate(tokens) if token in conjunctions]

    # Establish coordination pattern
    for idx in coord_indices:
        if idx > 0:
            out[idx, idx - 1] = 0.5
        if idx < len_seq - 1:
            out[idx, idx + 1] = 0.5

    # Ensure self-attention
    for i in range(len_seq):
        out[i, i] = 1.0

    # Normalize such that rows sum up to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Point to [SEP] if no attention
        else:
            out[row] /= out[row].sum()

    return "Conjunction and Coordination Pattern", out