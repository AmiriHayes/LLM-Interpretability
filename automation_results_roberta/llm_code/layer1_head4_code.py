import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention to the start token
    for i in range(1, len_seq - 1):
        out[i, 0] = 1  # Focus predominantly on the <s> token

    out[0, 0] = 1  # Self-attention for the start token
    out[len_seq - 1, 0] = 1  # End token attends to the start token

    # Ensure no row is all zeros by assigning minimal self-attention if needed
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1

    # Normalize the attention weights
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence-Level Start Token Dominance", out