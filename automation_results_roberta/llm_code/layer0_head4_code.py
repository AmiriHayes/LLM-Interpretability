from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function
def entity_and_verb_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    pivot_indices = [0, len_seq - 1]  # CLS and SEP tokens

    # Sample indices to pay attention to (entities and main verbs indices)
    for i, word in enumerate(words):
        if word in {"Lily", "needle", "mom", "shirt", "share", "sew", "sharp", "difficult", "smiled", "felt", "happy", "shared", "worked"}:
            pivot_indices.append(i + 1)  # Adjust for CLS token

    # Assign attention to specified indices
    for idx in range(1, len_seq - 1):
        if idx in pivot_indices:
            out[idx, idx] = 1  # Self attention for entities or verbs
        else:
            # Attending primarily to identified entities and verbs
            for pivot in pivot_indices:
                out[idx, pivot] += 1 / len(pivot_indices)

    out += 1e-4  # Prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize the output matrix by row

    return "Entity and Verb Focus Pattern", out