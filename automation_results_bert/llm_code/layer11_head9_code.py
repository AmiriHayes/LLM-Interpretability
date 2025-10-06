from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def end_of_sentence_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention on the last token (typically the period) for each token
    for i in range(len_seq):
        out[i, len_seq - 2] = 1  # len_seq - 2 for the second last position typically a period

    # Also, include self-attention for special tokens [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "End-of-Sentence Focus", out