from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

"""
This function generates a predicted attention pattern matrix for sentences encoded by a tokenizer. We hypothesize that the head focuses on temporal event resolution by prioritizing nearby temporal context tokens primarily associated with past or future narratives and significant actions that relate to the sentence's last predicate form, often resolving around verbs marked by boundary conditions.
"""
def temporal_event_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing and aligning sentence
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    significant_indices = []

    # Assume a simple context of verbs and specific words indicating temporal events
    temporal_tokens = {'before', 'after', 'while', 'as', 'when', 'until', 'since', 'then'}
    punctuation_set = {',', '.','?','!'}

    # Find significant token positions in the sentence
    for idx, token in enumerate(tokens):
        if any(punct in token for punct in punctuation_set) or token.lower() in temporal_tokens or 'VB' in token:
            significant_indices.append(idx)

    # Create attention on identified token indices, scoring inversely proportional to distance
    for i in range(1, len_seq-1):  # Exclude CLS and SEP
        for j in significant_indices:
            if i != 0 and i < len_seq-1 and j < len_seq:
                out[i, j] = 1 / (abs(i-j) + 1e-3)  # Small value to prevent division by zero

    # Normalize the attention matrix rows
    out = out / out.sum(axis=1, keepdims=True)

    # Assign some attention to the [CLS] token for every other word
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0  # Ensure no row in the attention map is entirely zero out

    return "Temporal Event Resolution", out