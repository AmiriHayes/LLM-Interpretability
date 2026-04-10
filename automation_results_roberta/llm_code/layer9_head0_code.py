import numpy as np
from transformers import PreTrainedTokenizerBase

def boundary_markers(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define special token indices for boundary markers
    cls_idx = 0  # index for <s> token
    sep_idx = len_seq - 1  # index for </s> token

    # Strong attention from each token to sentence boundary markers
    for i in range(len_seq):
        out[i, cls_idx] = 1.0 / len_seq  # normalize attention distribution
        out[i, sep_idx] = 1.0 / len_seq

    # Normalize attention pattern to ensure it adds up to 1 for each row
    row_sums = out.sum(axis=1)
    for i in range(len(out)):
        out[i] /= row_sums[i]

    return "Sentence Boundary Markers", out
