from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def topic_introduction_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks['input_ids'][0])
    out = np.zeros((len_seq, len_seq))

    # Assign the first content token same attention to the initial tokens
    # Treat first noun or first significant noun-related token as topic introducer
    content_indices = []
    for idx, input_id in enumerate(toks['input_ids'][0]):
        token_str = tokenizer.decode([input_id.item()]).strip()
        if len(token_str) > 1 and token_str.isalpha():  # Assuming longer tokens that are alphabetic as content tokens
            content_indices.append(idx)

    if content_indices:
        first_content_idx = content_indices[0]
        out[first_content_idx, :first_content_idx] = 1

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, where=row_sums != 0)

    return "Topic Introduction Emphasis", out