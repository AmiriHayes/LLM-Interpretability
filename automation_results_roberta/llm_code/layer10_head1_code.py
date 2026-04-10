import numpy as np
from transformers import PreTrainedTokenizerBase


def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention focuses primarily on the sentence boundary tokens
    for i in range(len_seq):
        if i == 0:  # [CLS] token attention
            out[i, -1] = 1.0 # Focus on [SEP]
        elif i == len_seq - 1:  # [SEP] token attention
            out[i, 0] = 1.0 # Focus on [CLS]

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default to attending to [SEP] if no focus is deduced

    return "Sentence Boundary Detection", out
