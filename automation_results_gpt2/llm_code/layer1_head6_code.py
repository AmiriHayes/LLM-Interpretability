import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_centering(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign highest attention to the first token, assuming it represents
    # the initial word of the sentence after tokenization
    for i in range(1, len_seq):
        out[i, 0] = 1
    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Initial Token Centering", out