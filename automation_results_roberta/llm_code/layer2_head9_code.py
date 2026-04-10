import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focusing heavy attention on sentence boundaries (CLS and SEP tokens)
    # CLS token representation is at index 0, SEP token representation is at the last index
    for i in range(len_seq):
        # Strong self-attention on CLS
        if i == 0:
            out[i, i] = 1.0
        # Attention on SEP
        elif i == len_seq - 1:
            out[i, i] = 1.0

    # For other tokens, ensure they have some minimal attention back to the SEP token
    for i in range(1, len_seq - 1):
        out[i, -1] = 1e-1 # Ensure there's minimal attention to SEP

    # Normalize the attention matrix
    row_sums = out.sum(axis=1, keepdims=True) + 1e-8 # Avoid division by zero
    out /= row_sums

    return "Sentence Boundary Recognition Pattern", out