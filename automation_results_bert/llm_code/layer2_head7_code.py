import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def conjunction_coherence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Example keywords indicating logical connections and coherence
    conjunctions = {'and', 'because', ',', '.'}
    conjunction_indices = []

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    for i, token in enumerate(tokens):
        if token.lower() in conjunctions:
            conjunction_indices.append(i)

    # Create attention patterns linking conjunctions to other tokens
    for idx in conjunction_indices:
        for j in range(len_seq):
            if j != idx:
                out[idx, j] = 1
                out[j, idx] = 1  # Making it bidirectional

    # Normalize the matrix rows
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    # Ensure no row is all zeros (adding self-attention bias)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0
            out /= out.sum(axis=1, keepdims=True)  # Re-normalize

    return "Conjunction and Coherence Pattern", out