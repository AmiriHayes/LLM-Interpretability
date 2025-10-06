from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def object_identification_and_connection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    token_to_index = {i: toks.input_ids[0][i].item() for i in range(len_seq)}
    # Set attention patterns for objects like needle, button, shirt, etc.
    keywords = set(["needle", "button", "shirt", "lily"])

    # Mark attention connections for each keyword instance.
    for i in range(1, len_seq - 1):
        if tokenizer.convert_ids_to_tokens(token_to_index[i]) in keywords:
            for j in range(1, len_seq - 1):
                # Focus on other keywords, except self
                if i != j and tokenizer.convert_ids_to_tokens(token_to_index[j]) in keywords:
                    out[i, j] = 1

    # Ensure each token at least attends to [SEP].
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize output
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Object Identification and Connection", out