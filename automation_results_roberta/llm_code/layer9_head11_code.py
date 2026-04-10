import numpy as np
from transformers import PreTrainedTokenizerBase

def boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str: 
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attend to <s> and </s> across all tokens significantly
    for i in range(len_seq):
        if i == 0: # <s> at position 0
            out[i, i] = 1
        elif i == len_seq - 1: # </s> at end
            out[i, i] = 1
        else:  # All tokens attend strongly to <s> and </s>
            out[i, 0] = 0.7
            out[i, len_seq - 1] = 0.6

    # Ensure no row is all zeros
    for row in range(len_seq): 
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Emphasizing Sentence Boundary Patterns", out