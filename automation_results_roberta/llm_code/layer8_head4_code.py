import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# This function captures attention on key nouns and verbs, leaning towards the end of the sentence.
def end_of_sentence_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize the last few (non-padding) tokens in each relevant segment.
    # The algorithm assumes no sentence ends in unwanted characters or artifacts beyond punctuation.
    # For uniformity in managing attention patterns, this only deals with positions before the EOS.
    num_end_tokens = 3  # how many of the tokens at the end to focus more on

    for i in range(1, len_seq - 1):  # Avoid CLS and EOS
        out[i, i] = 0.5  # Each word normally attends to itself, lower than emphasis
        for j in range(max(0, len_seq - num_end_tokens - 2), len_seq - 1):
            out[i, j] += 1.0 / num_end_tokens

    out[0, 0] = 1.0  # CLS
    out[-1, -1] = 1.0  # EOS

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize for attention distribution

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "End-of-Sentence Emphasis Pattern", out