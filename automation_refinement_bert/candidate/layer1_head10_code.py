import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Note: Consider using spacy only if necessary for token alignment, in this case, it should fit the tokenizer format

def long_distance_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis: Capture major long-distance dependencies such as subject-relation, verb-object.
    # Looking for patterns like subject-verb associations, e.g., he/speak, where long-distance tokens exhibit strong attention.

    for i in range(1, len_seq-1):  # Skip [CLS] and [SEP]
        if i+2 < len_seq:
            # Example simplistic long-distance: forward association
            out[i, i+2] = 0.5 
        if i-2 > 0:
            # Example simplistic long-distance: backward association
            out[i, i-2] = 0.5 

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0 

    # Normalize out matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Ensure the distribution is attention-like

    return "Long-distance Association Pattern", out