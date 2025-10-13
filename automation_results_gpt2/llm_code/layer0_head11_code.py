import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initiation_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign focus to the first real token after the initial token
    attention_index = 1

    for i in range(1, len_seq):
        out[i, attention_index] = 1.0

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Initiation Focus", out