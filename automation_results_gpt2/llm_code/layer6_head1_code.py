from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_hooks(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Most attention hooks to the first token after the special [CLS] token
    for i in range(1, len_seq-1):
        out[i, 1] = 1.0

    # Ensure every row has some attention value
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Initial Token Hooks", out