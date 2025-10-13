import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_protagonist(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign attention to the first token and the immediate few tokens within each sentence
    for i in range(min(5, len_seq)):  # assuming the first token to a few tokens after are the focus
        out[0, i] = 1

    # Normalize the attention such that the sum along the rows is 1
    out[0, :] = out[0, :] / out[0, :].sum()

    # Ensure each token has some attention to the end token to avoid zero rows
    for row in range(1, len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Start Protagonist Role", out