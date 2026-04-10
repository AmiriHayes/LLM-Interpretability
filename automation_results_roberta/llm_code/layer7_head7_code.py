import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_detection(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The pattern shows strong attention to start-of-sequence and end-of-sequence
    out[0, 0] = 1  # CLS token pays attention to itself
    out[-1, 0] = 1  # the end token pays attention to the start token
    out[-1, -1] = 1  # end token pays attention to itself

    for i in range(1, len_seq - 1):
        out[i, 0] = 1  # all tokens predominately paying attention to the CLS token

    # Ensure no row is entirely zeros to maintain valid attention.
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize across each row so that each row sums to 1.
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Detection", out