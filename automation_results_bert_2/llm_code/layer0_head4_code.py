import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def entity_action_connection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simulated behavior based on observation
    for i in range(1, len_seq - 1):
        out[i - 1, i] = 1  # Simulate some kind of incremental token attention
        if i < len_seq - 4:  # Simulate looking ahead to find important connections
            out[i, i + 2] = 1
            out[i, i + 3] = 0.5  # Partial attention to further tokens

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to simulate softmax or probabilistic behavior
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Entity and Action Connection Pattern", out