import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_word_contextualizer(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume the first word is always the focus and influences others
    out[0, 1:] = 1  # the first token attends to all others
    # The first token would attend more strongly, so we increase its value
    out[0, 0:len_seq] /= out[0, 0:len_seq].sum()

    # Make sure no row is exactly zeros by allowing self-attention for other tokens
    for i in range(1, len_seq):
        out[i, -1] = 1.0

    return "Initial Word Contextualizer", out