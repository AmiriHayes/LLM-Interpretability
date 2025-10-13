from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Predicting that sentences have a strong initial token attention pattern
    out[0, :] = 1.0 # CLS token attends to the entire sentence
    out[1, :] = 1.0 # First token attends to the entire sentence

    # Normalize attention by row sum
    row_sums = out.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    out = out / row_sums

    return "Sentence Start Attention Pattern", out