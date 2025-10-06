import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def needle_and_object_interaction(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming "needle" interaction is a critical focus and interaction with objects
    objects = ["needle", "button", "shirt"]
    object_indices = []
    needle_index = None

    words = sentence.split()
    for i, tok in enumerate(words):
        for obj in objects:
            if obj in tok.lower():
                object_indices.append(i)
            if "needle" in tok.lower():
                needle_index = i

    # Assign high attention to interactions with the needle
    if needle_index is not None:
        for obj_index in object_indices:
            if obj_index != needle_index:
                out[needle_index + 1, obj_index + 1] = 1
                out[obj_index + 1, needle_index + 1] = 1

    # Handle other interactions based on patterns observed in examples
    # Example: making an assumption of distributed focus for verbs and auxiliary verbs
    for i in range(1, len_seq-1):
        out[i, i] = 0.2  # Light self-attention

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Needle and Object Interaction Pattern", out