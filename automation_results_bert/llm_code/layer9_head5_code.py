import numpy as np
from transformers import PreTrainedTokenizerBase

def shared_object_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence and assume the second token as a keyword focus assumption 
    # based on patterns found in the data provided.
    for i in range(1, len_seq - 1):
        out[i, 1] = 1

    # Ensure CLS ([0]) and SEP [-1] alignments to fulfill the attention requirement
    out[0, 0] = 1   # CLS attends to itself
    out[-1, -1] = 1 # SEP attends to itself
    out[1, -1] = 0.5 # an arbitrary small constant to assure linkage to SEP from focus

    # Normalize the matrix to avoid zero rows which might represent unintended interpretations
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention to sum to 1 per row
    out = out / out.sum(axis=1, keepdims=True)

    return "Shared Object Focus Pattern", out