import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define the function

def possessive_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Align tokens from tokenizer with spaCy
    words = sentence.split()

    # Loop through token pairs to identify possessive relationships encoded as (owner, owned)
    for i, token in enumerate(words[:-1]):
        if token.lower() == "my" or token.lower() == "her" or token.lower() == "his" or token.lower() == "their":
            # Use heuristics to determine possession, usually follows with a noun
            if i + 1 < len(words):
                out[i, i + 1] = 1.0
        # Handle relationships like "Lily's shirt"
        if "'s" in token:
            out[i, i+1] = 1.0

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default attention to [SEP]

    return "Possessive and Ownership Relationships", out