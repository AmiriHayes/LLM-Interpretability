from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign high attention scores to the [CLS] and [SEP] tokens
    out[0, :] = 1.0  # The [CLS] token attends to everything
    out[:, 0] = 1.0  # Every token attends to the [CLS] token
    out[-1, :] = 1.0  # The [SEP] token attends to everything
    out[:, -1] = 1.0  # Every token attends to the [SEP] token
    # Normalize rows to ensure no row is completely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to focus attention
    return "Sentence Boundary Focus", out