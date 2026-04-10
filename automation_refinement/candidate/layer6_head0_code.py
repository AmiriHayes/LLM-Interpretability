import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def comma_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.input_ids[0]
    # Identify commas in the tokenized sentence
    commas = [i for i, tok in enumerate(tokens) if tokenizer.convert_ids_to_tokens(int(tok)) == ',']
    # Assign the association of commas with themselves and their surrounding context
    for comma_index in commas:
        # Assign higher attention to the token immediately preceding and succeeding a comma
        if comma_index > 0:
            out[comma_index, comma_index - 1] += 1
        if comma_index < len_seq - 1:
            out[comma_index, comma_index + 1] += 1
        # Self-attention on the comma itself
        out[comma_index, comma_index] += 1
    # Add some self-attention for CLS and SEP tokens to follow a general pattern
    out[0, 0] = 1
    out[-1, 0] = 1
    # Normalize the matrix
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, where=row_sums != 0)
    return "Comma Association", out