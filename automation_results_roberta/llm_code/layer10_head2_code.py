import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to the EOS and CLS tokens");
    out[0, -1] = 1.0
    out[-1, -1] = 1.0

    # Normalize each row to ensure no row is all zeros
    # This implicitly assumes there's a smaller bias toward sentence start
    for row in range(1, len_seq-1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Detection", out