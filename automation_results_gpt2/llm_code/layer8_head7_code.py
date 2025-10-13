import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention pattern
    # Every token attends primarily to the first content token
    for i in range(len_seq):
        out[i, 1] = 1  # Attend to the first non-special token (often a specific content start)

    # Ensure no row is all zeros by giving attention to the [CLS] or starting token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize row-wise

    return "Sentence Start Focus", out