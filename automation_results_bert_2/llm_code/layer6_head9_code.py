import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coord_subord_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize and guide based on provided patterns
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    cls_index, sep_index = 0, len_seq-1
    conj_subord_pairs = []

    # Detect conjunctions or subordination
    for i, word in enumerate(words):
        if 'and' in word or 'or' in word or 'because' in word:
            if i > 0:
                conj_subord_pairs.append((i-1, i))
            if i < len_seq-1:
                conj_subord_pairs.append((i, i+1))

    # Allocate attention based on conjunction and subordination rules
    for head, tail in conj_subord_pairs:
        out[head, tail] = 1
        out[tail, head] = 1

    for row in range(len_seq):
        # Ensure no row is all zeros, default to separator token attention
        if out[row].sum() == 0:
            out[row, sep_index] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention matrix
    return "Coordination and Subordination Detection", out

