import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    initial_phrase_word = words[0] if words else None

    initial_indices = []
    for i, tok in enumerate(toks.input_ids[0]):
        decoded_substring = tokenizer.decode([tok.item()])
        # Only consider tokens that contribute to the first word
        if initial_phrase_word and decoded_substring.strip().startswith(initial_phrase_word):
            initial_indices.append(i)

    for i in initial_indices:
        for j in range(len_seq):
            # Assign high attention to tokens sharing word origin as initial word
            if j in initial_indices:
                out[i, j] = 1.0

    # Normalize output matrix
    out += 1e-4  # Avoid division by zero
    out /= out.sum(axis=1, keepdims=True)  # Normalize by row

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Distribute remaining attention equally

    return "Initial Phrase Referential Attention", out