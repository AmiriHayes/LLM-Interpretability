import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def sentence_boundary_and_key_term_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to <s> and </s>
    out[0, 0] = 1.0  # <s> self-attention
    out[-1, 0] = 1.0  # </s> attends back to <s>

    # Key term focus: emulate the effect seen in data
    key_term_index = len_seq - 2  # Assume the second to last token is often a significant word
    out[key_term_index, 0] = 0.7  # Attention back to <s> as observed in examples
    out[key_term_index, -1] = 0.3  # Key term attends to </s> as observed

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Boundary and Key Term Focus", out