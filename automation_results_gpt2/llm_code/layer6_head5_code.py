import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_centralization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention value to the first content word for all words in the sentence
    # First content word here is considered after removing CLS token, which is usually the first token
    for i in range(1, len_seq):  # Start from token 1 assuming token 0 is CLS
        out[i, 1] = 1

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Sentence Start Centralization", out