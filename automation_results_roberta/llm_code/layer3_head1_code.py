import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def cls_sentence_embedding(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign highest attention to <s> (CLS token) across the board
    for i in range(len_seq):
        out[i, 0] = 1.0
    # Scale down to ensure it's a valid probability
    out = out / out.sum(axis=1, keepdims=True)
    return "CLS Sentence Embedding", out