import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define the function to predict the pattern

def coordinating_conjunction_role(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and get spans/span tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunction_indices = [i for i, token in enumerate(tokens) if token in {"and", "so","because", "but"}]

    # Assign attention patterns based on conjunctions
    for index in conjunction_indices:
        if 1 <= index < len_seq - 1:  # Ensure within valid range
            out[index, index-1] = 1    # Attention to the previous token
            out[index, index] = 1      # Self-attention
            out[index, index+1] = 1    # Attention to the next token

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign weak attention to [SEP]

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize across each row

    return "Coordinating Conjunction Semantic Role", out