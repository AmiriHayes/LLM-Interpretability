import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coord_completion(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a mapping from token position to position + 1 in the sentence for prediction
    for token_index in range(1, len_seq - 1):
        if token_index < len_seq - 2:  # Avoid going out of bounds
            out[token_index, token_index + 1] = 1

    # Preserve self-attention for the first and last token (CLS and SEP)
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coordination and Completion", out

