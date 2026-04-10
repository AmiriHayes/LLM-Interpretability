import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def punctuation_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    punctuation_indices = set()
    # Non-terminal punctuation, assume [CLS] = 0 and [SEP] = len_seq - 1
    for i in range(1, len_seq - 1):
        token = tokenizer.convert_ids_to_tokens(toks.input_ids[0][i].item())
        if token in {',', '.', ':', ';', '!', '?'}:
            punctuation_indices.add(i)

    # Model has a specific pattern targeting non-terminal punctuation
    for idx in punctuation_indices:
        for j in range(1, len_seq - 1):
            if j != idx:
                out[idx, j] = 1

    # Ensure self-attention for [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the attention across rows
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Non-terminal Punctuation Focus", out