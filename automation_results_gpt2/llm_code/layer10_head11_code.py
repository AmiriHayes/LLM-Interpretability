from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initial_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token typically has the highest attention weight.
    first_token_idx = 1  # Due to the CLS token at the start, our interest might be at index 1.

    # Focus on the first token primarily.
    for i in range(1, len_seq):  # Exclude the CLS token (index 0 if it exists, else start from 1)
        out[i, first_token_idx] = 1.0

    # Normalize so each row sums to 1; ensures valid attention distribution.
    for i in range(len_seq):
        if out[i].sum() == 0:
            out[i, -1] = 1.0  # Minimal attention to the last token if no other weight

    out = out / out.sum(axis=1, keepdims=True)  # Normalize across rows

    return "Sentence Initial Token Focus", out