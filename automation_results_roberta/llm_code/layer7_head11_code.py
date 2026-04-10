import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def prominence_anchor(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign prominent weights to the CLS token and last position of the sequence
    CLS_INDEX = 0
    SEP_INDEX = len_seq - 1

    for i in range(len_seq):
        out[i, CLS_INDEX] = 1  # High attention to CLS token
        out[i, SEP_INDEX] = 1  # High attention to [SEP] token or last token

    # Normalize the attentions
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize each row

    return "Prominence of Anchoring Tokens Pattern", out