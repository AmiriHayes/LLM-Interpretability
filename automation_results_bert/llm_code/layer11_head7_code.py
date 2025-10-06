import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_clause_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Define a simple attention pattern: focus on coordinating conjunctions
    for i, tok in enumerate(words):
        if tok in {',', 'and', 'but', 'because', 'so'}:
            out[i, i] = 1  # self-attention to conjunction or clause boundary indicators
            if (i + 1) < len_seq:  # focus on the next word 
                out[i, i + 1] = 1
            if (i - 1) >= 0:  # focus on the previous word to mark boundaries
                out[i, i - 1] = 1

    # Ensure no row is entirely zeros; default cls and sep behavior
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordinating Conjunction and Clause Boundary Role", out