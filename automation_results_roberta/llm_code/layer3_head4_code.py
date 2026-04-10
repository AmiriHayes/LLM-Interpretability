import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign strong attention from all tokens to the start token except special tokens
    for i in range(1, len_seq-1):
        out[i, 0] = 1.0

    # Ensure no row is all zeros: assign self-attention to special tokens if necessary
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    return "Sentence Start Attention Pattern", out