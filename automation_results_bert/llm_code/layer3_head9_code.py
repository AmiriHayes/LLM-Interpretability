from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to model the hypothesized pattern

def noun_phrase_completion(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Custom rules for noun-phrase completion attention pattern
    word_ids = toks.word_ids(batch_index=0)
    for i in range(len_seq):
        for j in range(len_seq):
            # Capture nouns or main objects and nearest connected word
            if word_ids[i] is not None and word_ids[j] is not None:
                # Set higher attention to potential paired noun and related word
                out[i, j] = 1 if word_ids[i] == word_ids[j] else 0.1

    # Ensure at least one attention value per row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Noun-Phrase Completion", out