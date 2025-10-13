from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def initial_token_role_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Emphasize attention on the very first token (excluding [CLS])
    for i in range(1, len_seq - 1):
        out[i, 1] = 1  # Maximum attention on the initial token
    # Ensure no row is all zeros (either make the first or last column non-zero)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Initial Token Role Emphasis", out