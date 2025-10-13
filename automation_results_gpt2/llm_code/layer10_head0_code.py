import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set the first content token (ignoring CLS token) to have maximum attention weight on others
    for i in range(1, len_seq - 1):
        out[1, i] = 1  # Assuming the first significant token has major attention
    # Normalize the attention matrix by row to ensure valid attention distribution
    row_sums = np.sum(out, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.  # Avoid division by zero
    out = out / row_sums
    return "Initial Token Attention Pattern", out