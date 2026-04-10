from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_centered_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a simple punctuation list
    punctuation_tokens = {".", ",", ":", "'", '"'}

    # Fill attention around punctuation marks
    for i in range(1, len_seq - 1):
        token_id = toks.input_ids[0][i].item()
        token_str = tokenizer.decode([token_id])

        if token_str in punctuation_tokens:
            # distribute attention from punctuation
            out[i] = 1.0 / (len_seq - 1)
            out[i, i] = 1  # Add self-attention for punctuation
            out[i] = out[i] / out[i].sum()  # Normalize row

    # Ensure no row is all zeros by giving any token attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Punctuation Centered Attention", out