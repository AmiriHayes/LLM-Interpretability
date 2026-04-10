import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_group_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention primarily on the first non-punctuation content token and its nearest content tokens
    for i in range(1, len_seq-1):
        # Assume the first non-punctuation token has the highest importance
        # Attention decreases as we move away from it
        attention_weight = max(0, 1 - 0.1 * (i - 1))
        out[i, 1] = attention_weight

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Token Group Focus", out