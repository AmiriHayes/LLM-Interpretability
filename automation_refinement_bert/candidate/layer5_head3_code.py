import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def comma_based_phonological_phrase_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into phonological phrases based on comma presence
    delim_points = [i for i, token in enumerate(toks.tokens()) if token == ',']
    delim_points = [0] + delim_points + [len_seq - 1]  # Include [CLS] and [SEP] as bounds

    # Create links within chunks divided by commas
    for start, end in zip(delim_points[:-1], delim_points[1:]):
        for i in range(start, end + 1):
            out[i, start:end+1] = 1
            out[start:end+1, i] = 1

    # Ensure no row is all zeros (handling [CLS] and [SEP])
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Comma-Based Phonological Phrase Linkage", out