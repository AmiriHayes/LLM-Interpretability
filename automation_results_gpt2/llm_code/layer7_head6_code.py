import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_word_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Focus pattern where attention is given mostly to the first word
    main_focus_idx = 1  # Assuming the first word token after CLS token is index 1
    out[:, main_focus_idx] = 1
    # Normalize to sum to 1 across each row
    for row in range(len_seq):
        if out[row].sum() > 0:
            out[row] /= out[row].sum()
    return "Initial Word Focus", out