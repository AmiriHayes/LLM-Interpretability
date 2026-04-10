import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assigning strong focus to the first non-special token, assuming token 0 is special (like CLS)
    for j in range(1, len_seq):
        out[1, j] = 1
    # Normalize each row by dividing by its sum (to simulate attention probabilities)
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, where=row_sums!=0)
    return 'Initial Token Focus', out