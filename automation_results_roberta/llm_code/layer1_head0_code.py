import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function to predict sentence boundary attention pattern

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # <s> (CLS) token attends mainly to itself.
    out[0, 0] = 1

    # </s> (SEP) token attends mainly to itself.
    if len_seq > 1:
        out[len_seq - 1, len_seq - 1] = 1

    # All other tokens focus a majority of attention on the <s> token, similar to global attention patterns
    for i in range(1, len_seq - 1):
        out[i, 0] = 1

    # Normalize rows to ensure all probabilities sum to 1, avoiding division by zero by adding a small epsilon
    out = (out.T / np.sum(out, axis=1)).T

    return "Sentence Boundary Attention", out
