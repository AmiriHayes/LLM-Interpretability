from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def typographical_influence_minimization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # We hypothesize that the head dampens influence from certain punctuation marks
    # like ',' and possibly attends more heavily to nouns related to the current focus.
    # This code tries to capture that by halving attention to tokens following a comma.
    for i in range(1, len_seq - 1):
        if toks.input_ids[0][i - 1] == tokenizer.convert_tokens_to_ids(','):
            # Reduce influence after commas
            continue
        out[i, i] = 1  # Mainly self-attend
        # Distribute some attention there
        if i + 1 < len_seq:
            out[i, i + 1] = 0.5

    # Ensure cls and eos tokens have self-attention
    out[0, 0] = 1
    out[-1, -1] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention weights in each row
    out += 1e-9  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Typographical Influence Minimization", out