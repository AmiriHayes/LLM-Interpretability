import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def verb_and_preposition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    """
    Function to predict an attention pattern focusing on verbs and prepositions in layer 6, head 2.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming we can identify verbs and prepositions positions for demonstration
    # In practice, consider using an NLP library like spaCy for more precise positions
    words = sentence.split()
    verb_indices = [i + 1 for i, word in enumerate(words) if word in {'found', 'knew', 'wanted', 'went', 'said', 'can', 'smiled', 'shared', 'was', 'finished', 'felt', 'had'}]
    preposition_indices = [i + 1 for i, word in enumerate(words) if word in {'in', 'with', 'to', 'and', 'because', 'for', 'on', 'after', 'together'}]

    for vi in verb_indices:
        for pi in preposition_indices:
            out[vi, pi] = 1
            out[pi, vi] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Verb and Preposition Attention", out