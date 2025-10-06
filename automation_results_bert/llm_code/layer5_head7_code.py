import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def verb_object_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A heuristic approach to mimic verb-object relationships
    words = sentence.split()
    verb_indices = []
    object_indices = []

    # Simple verb and object identification based on POS tagging
    for i, word in enumerate(words):
        # Using heuristic checks, these would be POS tagging in practice
        if word in {"found", "knew", "wanted", "said", "share", "shared", "thanked", "felt", "sewed", "fix", "helping", "playing"}:  # indicative of verbs
            verb_indices.append(i+1)
        elif word in {"needle", "room", "button", "shirt", "mom", "it", "them", "each", "other"}:  # indicative of objects
            object_indices.append(i+1)

    # Assign high attention to verb-object pairs
    for v_i in verb_indices:
        for o_i in object_indices:
            out[v_i, o_i] = 1

    # Ensure no row has all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    # Normalize the attention matrix
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Verb-Object Relationship Recognition", out