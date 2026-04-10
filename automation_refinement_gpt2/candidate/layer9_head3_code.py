from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention heavily on the first non-special token
    first_non_special_token_index = 1  # Assuming token at index 1 is the first non-special token like 'The', 'She', etc.

    # Assign a high attention score to the first non-special token
    for i in range(1, len_seq):
        out[i, first_non_special_token_index] = 1

    # Self-attention for the CLA and EOS tokens
    out[0, 0] = 1  # Self-attention for CLS
    out[-1, -1] = 1  # Self-attention for EOS

    # Normalize the attention matrix by row
    for i in range(len_seq):
        row_sum = out[i].sum()
        if row_sum > 0:
            out[i] /= row_sum

    return "Sentence-Initial Token Focus", out