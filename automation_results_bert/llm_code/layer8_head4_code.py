import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coordination_detection_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Sample pattern based on example data
    words = sentence.split()
    conjunction_indices = [i for i, word in enumerate(words) if word.lower() in {"and", "because", ",", "so"}]

    for idx in conjunction_indices:
        # Suppose conjunctions point to the following word, which often appears to govern/alter meaning
        if idx + 1 < len(words):
            out[idx + 1, idx + 2] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Coordination and Conjunction Detection Pattern", out