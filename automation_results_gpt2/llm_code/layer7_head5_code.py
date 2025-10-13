import numpy as np
from transformers import PreTrainedTokenizerBase

def focus_on_initial_token(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Focus all attention from each token to the initial token
    for i in range(1, len_seq):
        out[i, 1] = 1  # 1 accounts for [CLS]

    # Ensure at least 1 attention weight in each row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 1] = 1  # redirect to the initial token if the token does not attend anywhere

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize to sum to 1
    return "Focus on Initial Token Pattern", out