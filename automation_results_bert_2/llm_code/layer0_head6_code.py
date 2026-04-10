import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to encode observed attention pattern

def repetitive_entity_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing the sentence
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Dictionary to track last seen indices of tokens (ignoring special tokens)
    last_seen = {}

    for i, token in enumerate(tokens):
        if token not in ["[CLS]", "[SEP]"]:
            # If we've seen this token before, match to previous occurrence
            if token in last_seen:
                last_idx = last_seen[token]
                out[i, last_idx] = 1
                out[last_idx, i] = 1
            last_seen[token] = i

    # Ensuring no row in the matrix is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Repetitive Entity Attention", out