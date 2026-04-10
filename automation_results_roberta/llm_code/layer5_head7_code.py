from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize the sentence start, end and punctuation
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.1  # small attention to CLS
        out[i, len_seq - 1] = 0.5  # moderate attention to SEP
        if sentence[i].strip() in ",." and i < len_seq - 1:
            out[i, i+1] = 0.3  # add attention to next token if punctuation

    # Assign strong attention to START and END markers
    out[0, 0] = 0.9  # CLS token
    out[len_seq - 1, len_seq - 1] = 0.9  # SEP token

    # Normalize
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] += 1e-4
        out[row] = out[row] / out[row].sum()

    return "Sentence Boundary Emphasis", out