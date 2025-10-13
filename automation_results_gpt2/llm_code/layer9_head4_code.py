import numpy as np
from transformers import PreTrainedTokenizerBase, GPT2Tokenizer
from typing import Tuple

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Each token primarily pays attention to the first content word
    # The first non-special token in the sentence is typically considered the attention focus
    focus_token_index = 1  # Assuming it starts right after CLS (index 0), can adjust if needed.

    for i in range(1, len_seq):
        out[i, focus_token_index] = 1  # Each token pays attention to the identified focus word

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Start Focus", out