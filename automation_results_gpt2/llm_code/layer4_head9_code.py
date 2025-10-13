import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention towards the initial tokens of clauses/sentences
    for i in range(1, len_seq):
        out[i, 1] = 1  # Assume the first actual token after CLS gets highest attention

    # Ensure attention distribution consistency
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # default to padding token if no attention is placed
        out += 1e-4  # Avoid division by zero
        out = out / out.sum(axis=1, keepdims=True)  # Normalize by rows

    return "Sentence Start Emphasizing Pattern", out