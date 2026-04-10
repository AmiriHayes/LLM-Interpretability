import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def coordinate_range_encoding(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    coordinates = False

    # A heuristic to find coordinates or ranges
    for i in range(1, len_seq-1):
        token_id = toks.input_ids[0, i].item()
        token_str = tokenizer.decode(token_id)

        # Check for simple pattern that could indicate a coordinate or numerical range
        # e.g., patterns like '( x , y )' or 'x - y'
        if token_str in {',', '-', ')', '('} or token_str.isdigit() or token_str.isalpha():
            coordinates = True
            out[i-1, i] = 0.8
            out[i, i-1] = 0.8

    # Normalize the out matrix
    if coordinates:
        row_sums = out.sum(axis=1, keepdims=True)
        out = out / row_sums

    return "Coordinate and Range Encoding", out