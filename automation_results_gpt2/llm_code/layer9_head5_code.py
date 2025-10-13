import numpy as np
from transformers import PreTrainedTokenizerBase

def focus_on_initial_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention weights to the first non-CLS token (index 1)
    for i in range(1, len_seq):
        out[i, 1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Focus on Sentence Initial Token', out