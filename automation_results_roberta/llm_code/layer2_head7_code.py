import numpy as np
from transformers import PreTrainedTokenizerBase

def cls_token_domination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # CLS Token Dominates the attention
    for i in range(len_seq):
        out[i, 0] = 1  # Every token attends to the CLS token

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize each row
    return "CLS Token Domination Pattern", out