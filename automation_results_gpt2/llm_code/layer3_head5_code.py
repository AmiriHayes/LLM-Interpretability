import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The model seems to focus on the start of the sentence and words closely following it
    # Assign higher attention to the first couple tokens
    for i in range(1, min(5, len_seq-1)):  # Ensure we don't exceed sequence bounds
        out[0, i] = 1.0  # Attention from the first token to other nearby tokens

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign some minimal attention where needed

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Start Attention Pattern", out