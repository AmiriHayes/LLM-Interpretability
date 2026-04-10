from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to analyze list element grouping in a sentence
def list_element_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = sentence.split()

    # List of separators commonly used alongside list elements
    separators = [',', 'and', 'or']

    # Match tokens with list elements and separators
    idx_to_token = {i: token for i, token in enumerate(tokens)}

    # Identify positions of list elements and link them using attention
    for i, token in idx_to_token.items():
        if token in separators:
            if i-1 >= 0:  # Link previous element to current separator
                out[i-1, i] = 1
            if i+1 < len_seq:  # Link current separator to next element
                out[i, i+1] = 1

    # Ensure no row is all zeros by defaulting empty ones to attend to [SEP] and [CLS]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "List Element Grouping Pattern", out