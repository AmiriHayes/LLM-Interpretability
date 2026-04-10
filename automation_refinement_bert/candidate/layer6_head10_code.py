python
import numpy as np
from transformers import PreTrainedTokenizerBase

def comma_centric_coordination_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Attention rule: focus primarily on the token following commas
    for i in range(1, len_seq-1):
        token_word = words[i-1]
        if token_word == ',':
            # Assign attention weight to the following token
            out[i, i+1] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Comma-Centric Coordination Attention Pattern", out