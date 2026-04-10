import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initial_token_linking(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # This head seems to consistently link the first token to several subsequent tokens
    # For every token in the sequence, apply attention from the first token
    for i in range(1, len_seq):  # Start from 1 to skip focusing on [CLS] or initial special tokens
        out[1, i] = 1.0  # Pay attention from first token to all others

    # Ensure all rows are non-zero to prevent division issues during normalization
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize each row
    out += 1e-4  # Avoid division by zero, add small constant
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to get the predicted attention matrix
    return "Sentence Initial Token Linking", out