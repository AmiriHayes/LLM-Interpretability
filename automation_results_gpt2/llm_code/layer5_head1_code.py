from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def first_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention from all tokens to the first actual token
    for i in range(1, len_seq-1):  # Ignoring special tokens usually at 0 and len_seq-1
        out[i, 1] = 1.0  # Attention directed towards the first token after the special [CLS]

    # Special token [CLS] self-attends
    out[0, 0] = 1.0

    # Ensure every row sums to 1 by normalizing
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "First Token Emphasis", out