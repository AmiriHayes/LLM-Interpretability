import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def positional_and_content_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus on sentence start, end and key content positioning
    for i in range(len_seq):
        if i == 0 or i == len_seq - 1:  # CLS and SEP tokens
            out[i, i] = 1
        else:
            # Focus distribution as observed in active examples
            if i == 1:  # Typically important word at the beginning
                out[i, 0] = 0.7
                out[i, i] = 0.3
            elif i == len_seq-2:  # Prioritize end tokens before EOS
                out[i, len_seq-1] = 0.7
                out[i, i] = 0.3

    # Ensure no row is completely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Positional and Content Focus Pattern", out

