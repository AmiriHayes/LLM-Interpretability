from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def intensity_emphasis_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Key words or tokens often determine emphasis: let's say, highly emphasized with high frequencies or intensity
    # This model seems to repeat the strongest attention value several times on key phrase pairs: "of | of", "to | to"

    important_tokens = {'and', 'to', 'of', 'for', 'with'}  # commonly observed in data

    # Assign strong attention values for pairs where intensity seems max
    for i in range(1, len_seq-1):
        token = toks.tokens()[0][i]

        if token in important_tokens:
            out[i, i] = 1.0

    # Emphasizing sentence start and end (CLS and SEP which usually have separate attention)
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention pattern across tokens
    out = out / out.sum(axis=1, keepdims=True)
    return "Intensity or Emphasis Attention Pattern", out