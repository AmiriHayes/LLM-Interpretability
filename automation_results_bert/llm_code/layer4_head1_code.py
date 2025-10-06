import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coordination_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    key_phrases = ['and', ',', '.', ';', '?', 'because']
    key_indices = []

    # Identify positions of key phrases
    tokens = [tokenizer.decode(toks.input_ids[0][i].item()) for i in range(len_seq)]
    for i, token in enumerate(tokens):
        if token in key_phrases:
            key_indices.append(i)

    # Set attention for key events or actions
    for idx in key_indices:
        for jdx in range(1, len_seq - 1):  # avoid [CLS] and [SEP]
            out[jdx, idx] = 1
            out[idx, jdx] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out = out + 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Coordination and Alignment of Key Events or Actions Pattern", out