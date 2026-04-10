import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def mathematical_operation_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming a basic pattern focusing attention between mathematical terms
    for i in range(len_seq):
        token = toks.input_ids[0][i].item()
        # Here we would focus on detecting math-related tokens
        if token in [1010, 1012, 1013, 1015, 1023]:  # Assumed token IDs for math ops
            for j in range(len_seq):
                out[i, j] = 1

    # Assign self-attention to [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix by rows
    out += 1e-5  # A small bias to avoid divide by zero
    row_sums = out.sum(axis=1, keepdims=True)
    out /= row_sums

    return "Mathematical Operation Detection Pattern", out