from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initialization_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Based on the data pattern, the first token receives the majority of attention.
    # Therefore, we simulate this by setting the first token row to have a consistent pattern.
    out[0, :] = 1 / float(len_seq)

    # For each row except the first and last, emulate attention focusing primarily on the first token
    for i in range(1, len_seq):
        out[i, 0] = 0.9  # Assume 90% of attention goes to the first token
        out[i, i] = 0.1  # Assume a token also minimally points to itself

    # Normalize each row to ensure it sums to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence-Level Initialization Pattern", out