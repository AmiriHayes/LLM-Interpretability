from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function

def sentence_start_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention from every token to the sentence start token
    for i in range(1, len_seq-1):
        out[i, 0] = 1

    # Ensure attention to the sentence end, especially to <cls> where no other attention happens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention scores by row
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    # Return the pattern name and the predicted attention matrix
    return "Sentence Start Emphasis", out