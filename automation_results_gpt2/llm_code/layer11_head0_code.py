from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def positional_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Introduce a strong bias towards the first token and then decrease attention strength
    for i in range(1, len_seq-1):
        out[i, 0] = 1 / (i + 1)  # Decreasing attention on the [CLS]/first token pattern

    # Ensure each token has a self-attention at least
    np.fill_diagonal(out, 0.1)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize out so each row sums to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Positional Attention Pattern", out