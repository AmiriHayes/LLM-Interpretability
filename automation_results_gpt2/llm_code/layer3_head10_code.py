import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def subject_based_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume the first word is usually the subject (simplification)
    # Set self-attention for the first token (often the main subject or start of a subject)
    out[1, 1] = 1  # Typically the first substantive word

    # Implement initial sequence attention from the subject's position
    for i in range(2, len_seq):
        if i < 6:  # Assuming the subject has initial dominion up to 5 words
            out[1, i] = 1 / (i - 1)  # Slight decay as distance increases

    # Ensure the sum of attention scores in each row is not zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention scores per row

    return "Subjects have self-attention and initial sequence attention", out
