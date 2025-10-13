from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Ensure that the first token attends to all tokens with gradually decreasing attention
    for col in range(len_seq):
        out[0, col] = 1 / (col + 1)
    # Normalize to sum to 1 across the row
    out[0] = out[0] / np.sum(out[0])
    # Ensure CLS and EOS have self attention
    out[0, 0] = 1 # CLS token attention to self
    out[-1, -1] = 1 # EOS token attention to self
    # Ensure there is at least one non-zero value for every token via self-attention
    for row in range(1, len_seq-1):
        out[row, row] = 1.0
    return "Initial Token Dominance Pattern", out