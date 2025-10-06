import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_transition_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Conjunctions and transition words
    conjunctions = {'and', 'but', 'because', 'so'}
    transition_indices = []

    for idx, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id]).strip()
        if token.lower() in conjunctions:
            transition_indices.append(idx)

    for idx in transition_indices:
        # Focus on the transition word itself
        out[idx, idx] = 1
        # If possible, attend to the previous token as well to establish connection
        if idx > 0:
            out[idx, idx - 1] = 1

    # Ensure non-transition words have at least attention to themselves or to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction and Sentence Transition Pattern", out