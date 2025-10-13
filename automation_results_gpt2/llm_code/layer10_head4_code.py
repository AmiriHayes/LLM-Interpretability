from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_opening_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    opening_token_index = 1
    for i in range(1, len_seq):
        out[opening_token_index, i] = 1
        # Ensure the last token (e.g., period or special tokens) doesn't have zero attention
        out[i, -1] = 0.1
    out += 1e-4  # Offset to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention weights
    return "Sentence-opening Token Focus", out