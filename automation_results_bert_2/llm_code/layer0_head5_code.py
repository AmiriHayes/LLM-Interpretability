from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def punct_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tok_ids = toks['input_ids'].squeeze().tolist()

    for i in range(1, len_seq):
        # Look to the immediate right, if possible
        if i + 1 < len_seq:
            out[i, i + 1] = 1

        # If token is period, give attention to self
        if tok_ids[i] == tokenizer.sep_token_id or tok_ids[i] == tokenizer.cls_token_id:
            out[i, i] = 1
            if i > 1:
                out[i, i - 1] = 1

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize
    return "Short Range Dependency and Punctuation Linking", out