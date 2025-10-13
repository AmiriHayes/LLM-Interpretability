import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The primary attention is focused on the first token of the sentence
    # We assume that tokens[1] corresponds to the first non-special token in the sequence
    for i in range(1, len_seq):
        out[i, 1] = 1

    # Ensure non-special tokens attend to themselves
    for i in range(1, len_seq - 1):
        out[i, i] = 0.5

    # Handle CLS and SEP-like tokens, assuming that token 0 is special (e.g., CLS)
    out[0, 0] = 1  # CLS-like attention token attends to itself
    out[len_seq - 1, len_seq - 1] = 1  # SEP-like attention token attends to itself

    # Normalize attention scores across each row
    out += 1e-4  # To prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Focus Pattern", out