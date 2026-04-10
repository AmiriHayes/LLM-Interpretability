import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function to identify the hypothesis pattern

def commas_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize and initialize attention matrix
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence by ' ', maintain a map of token indices
    words = sentence.split()
    token_to_word_map = {i: word for i, word in enumerate(words)}

    # Identify comma positions in tokens
    comma_indices = [i for i, word in token_to_word_map.items() if ',' in word]

    # Predict attention pattern
ddif len(comma_indices) > 1:
        for i in range(1, len(comma_indices)):
            curr_comma = comma_indices[i]
            prev_comma = comma_indices[i-1]
            out[prev_comma+1, curr_comma+1] = 1

    # Ensure attention isn't zero for any row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention values
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Commas Grouping Attention", out