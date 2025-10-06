import numpy as np
from transformers import PreTrainedTokenizerBase
import re

# Function for identifying coordinating conjunctions and their relations

def coordinating_conjunctions(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence
    tokens = toks.tokens()

    # Define coordinating conjunctions
    coordinating_conjs = {"and", "but", "or", "nor", "for", "yet", "so"}

    # Create attention-like patterns around coordinating conjunctions
    for i, token in enumerate(tokens):
        if token in coordinating_conjs:
            # Link current conjunction to its surrounding verbs
            # Check previous and next tokens for verbs or relevant content words
            if i > 0:  # Prevent out of bounds
                out[i, i - 1] = 1  # Previous token
            if i + 1 < len(tokens):  # Prevent out of bounds
                out[i, i + 1] = 1  # Next token

            # Propagate back and forth for phrasal verbs/conjunctions
            j = i - 2
            while j >= 0 and tokens[j] not in coordinating_conjs:
                out[j, i] = 0.5
                j -= 1
            j = i + 2
            while j < len(tokens) and tokens[j] not in coordinating_conjs:
                out[j, i] = 0.5
                j += 1

    # Normalize matrix rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] = out[row] / out[row].sum()

    return "Coordinating Conjunctions and Related Phrasal Verbs", out