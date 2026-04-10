import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def conjunction_sentence_transition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    conjunctions = {'and', 'but', 'or', 'so', 'for', 'nor', 'yet', 'because'} # Basic conjunctions

    # Mark attention for conjunctions
    for idx, word in enumerate(words):
        word = word.lower()
        if word in conjunctions:
            out[idx+1, :] = 1.0 / len_seq
            out[idx+1, 0] = 0.10 # Slight higher attention to <s> token

    # Normalize
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    # Provide a slightly higher focus for transitions
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Conjunction and Sentence Transition Detection', out