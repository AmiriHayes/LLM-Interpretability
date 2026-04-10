from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def sentence_start_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention of each token to the start of the sentence
    for i in range(len_seq):
        out[i, 0] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Start Emphasis", out