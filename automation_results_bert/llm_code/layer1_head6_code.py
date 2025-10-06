from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Identify conjunctions and coordinate items in a simple way
    for idx, tok in enumerate(words):
        if tok in {"and", "or", "but", "so"}:
            # Coordinate the word immediately before and after the conjunction
            if idx > 0:
                out[idx, idx - 1] = 1  # Link conjunction to the previous word
            if idx < len(words) - 1:
                out[idx, idx + 1] = 1  # Link conjunction to the next word

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Pay attention to [SEP] to ensure valid attention weights

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Conjunction Coordination Pattern", out