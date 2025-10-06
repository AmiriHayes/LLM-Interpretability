import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def collaboration_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence to match indices
    tokens = sentence.split()
    token_ids = {i: token for i, token in enumerate(tokens)}

    # Target words related to repeated action or collaboration
    keywords = {'share', 'shared', 'sharing', 'helping', 'together', 'needle'}

    # Loop through the tokens to set attention
    for i, token in token_ids.items():
        # Match the token against collaboration-related keywords
        if token in keywords:
            # Assign higher attention score to next token to simulate collaboration pattern
            if i + 1 < len_seq:
                out[i, i+1] = 1  # Simulate attention to the next token

    # Add attention to the sentence boundaries
    out[0, 0] = 1  # CLS attention to itself
    out[-1, -1] = 1  # SEP attention to itself

    # Ensure every token has some attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Repeated Action and Collaboration Pattern", out