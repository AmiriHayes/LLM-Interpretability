from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Function to predict the attention pattern for conjunction coordination

def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenization
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify coordinates from tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    coord_indices = [i for i, tok in enumerate(tokens) if tok.startswith('and') or tok.startswith(',')]

    # Assign high attention scores between each coordinate set and its arguments.
    for coord_index in coord_indices:
        # Assign attention in the range before and after the conjunction or comma
        for i in range(1, coord_index):
            for j in range(coord_index + 1, len_seq - 1):
                out[i, j] = 1
                out[j, i] = 1

    # Normalize the attention matrix by row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out /= out.sum(axis=1, keepdims=True)

    return "Conjunction Coordination", out