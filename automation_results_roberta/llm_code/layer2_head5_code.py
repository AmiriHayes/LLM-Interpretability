from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign attention to the <s> token
    for i in range(1, len_seq):  # Excluding [CLS] itself
        out[i, 0] = 1  # All tokens attend mainly to the <s> token

    # Ensure each token attends slightly to itself
    for i in range(len_seq):
        out[i, i] += 1e-3  # Light self-attention

    # Normalize attention by row
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Focus Pattern", out