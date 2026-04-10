import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def sentence_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign attention to the start of the sentence
    for i in range(1, len_seq - 1):
        out[i, 0] = 1  # Most tokens attend heavily to <s>

    # Ensure that <s> and </s> tokens attend to themselves
    out[0, 0] = 1  # <s> token attends to itself
    out[-1, -1] = 1  # </s> token attends to itself

    # Ensure no row is all zeros for valid probability distribution
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix so each row sums to 1
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start and Structural Pattern", out