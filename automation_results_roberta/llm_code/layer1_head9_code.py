import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function to capture the sentence boundary attention pattern

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention to <s> and </s> tokens
    for i in range(len_seq):
        out[i, 0] = 1  # <s> token at position 0
        out[i, -1] = 1 # </s> token at the last position

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundary Attention", out