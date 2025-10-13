import numpy as np
from transformers import PreTrainedTokenizerBase
def focus_on_initial(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Focus on the first real token (position 1) for all tokens except the special tokens.
    for i in range(1, len_seq - 1):
        out[i, 1] = 1
    # Ensure starting and ending positions attend to themselves
    out[0, 0] = 1
    out[-1, -1] = 1
    # Normalize to avoid all-zero rows and divide by sum to ensure probabilities
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] /= out[row].sum() if out[row].sum() != 0 else 1
    return "Sentence Initial Token Clustering Pattern", out