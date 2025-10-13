import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def clarify_self_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize self and immediate local context
    for i in range(len_seq):
        out[i, i] = 1  # Strong self-attention
        if i-1 >= 0:  # Context before
            out[i, i-1] = 0.5
        if i+1 < len_seq:  # Context after
            out[i, i+1] = 0.5

    # Ensure no row is all zeros by attending to SEP or last token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Self-emphasis and Local Context Clarification", out