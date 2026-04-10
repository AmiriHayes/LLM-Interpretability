from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def infix_fragment_repetition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Find repeated fragments
    unique_fragments = {}
    for i, word in enumerate(words):
        fragment = word
        if fragment in unique_fragments:
            unique_fragments[fragment].append(i+1)
        else:
            unique_fragments[fragment] = [i+1]

    # Highlight attention for repeated fragments
    for indices in unique_fragments.values():
        if len(indices) > 1:
            for ind in indices:
                out[ind, indices] = 1

    # CLS and SEP self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Infix Fragment Repetition Attention", out
