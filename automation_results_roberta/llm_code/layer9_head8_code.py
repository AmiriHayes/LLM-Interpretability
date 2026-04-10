import numpy as np
from transformers import PreTrainedTokenizerBase
def sentence_closure_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign high attention to the end token
    end_token_index = len_seq - 1
    for i in range(1, len_seq):
        out[i, end_token_index] = 1.0
    # Normalize the matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] += 1e-4  # Avoid division by zero
        out[row] /= out[row].sum()  # Normalize each row
    return "Sentence Closure Pattern", out