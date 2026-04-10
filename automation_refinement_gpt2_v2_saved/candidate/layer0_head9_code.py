import numpy as np
from transformers import PreTrainedTokenizerBase

def header_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[0] = 1  # Emphasize the importance of the header token or first index
    return "Header Token Emphasis", out