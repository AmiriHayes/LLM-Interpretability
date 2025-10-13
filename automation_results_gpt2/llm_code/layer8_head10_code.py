import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def focus_on_sentence_starters(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Typically, the attention is heavy on the initial token in a sentence
    # Assign high attention weights to the first substantial token after CLS
    first_token = 1  # Assuming the first token is CLS 
    out[first_token, :] = 1

    # Ensure each word attends to the start word
    for i in range(1, len_seq - 1):
        out[i, first_token] = 1

    # Normalize the matrix row-wise
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] /= out[row].sum()  # Normalize to create valid attention probabilities

    return "Focus on Sentence Starters", out