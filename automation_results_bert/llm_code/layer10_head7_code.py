from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def coordination_conjunction(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the token ids that correspond to common conjunctions
    conjunction_ids = []
    for idx, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id]).strip()
        if token in {"and", "but", "or", ",", "because", "so", ";"}:
            conjunction_ids.append(idx)

    # Assign attention by connecting conjunctions with neighboring words
    for conj_idx in conjunction_ids:
        if conj_idx > 0:
            out[conj_idx, conj_idx - 1] = 1  # connect to previous
        if conj_idx < len_seq - 1:
            out[conj_idx, conj_idx + 1] = 1  # connect to next

    # Ensure CLS and SEP have self-attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to ensure numerical stability
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination and Conjunction Focus", out