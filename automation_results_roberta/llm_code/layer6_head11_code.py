import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def period_position_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The pattern suggests attention to sentence boundaries and terminal punctuation
    # Assign highest scores for end of sentence and start
    for i, token_id in enumerate(toks.input_ids[0]):
        if i == 0 or i == len_seq - 1:  # Strong attention to [CLS] and [SEP]
            out[i, i] = 1
            continue
        decoded_tok = tokenizer.decode(token_id)
        if decoded_tok == '.':  # Special focus on periods
            out[i, i] = 0.5
            out[i, 0] = 0.5  # Strong attention also to [CLS] token
            out[i, len_seq - 1] = 0.5  # and to [SEP] token
    # Normalize
    out += 1e-4  # Prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Ensure each row sums to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure no row is completely without attention
    return "Period Position Recognition and Terminal Punctuation Focus", out