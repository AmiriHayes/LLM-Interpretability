from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def sentence_initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Focus attention on the first token after the CLS token, excluding special tokens
    out[1, 1:] = 1  # Set all other tokens to focus specifically on the first real token
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default to [SEP] if not focusing on any token
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to create a valid attention pattern
    return "Sentence Initial Token Focus", out