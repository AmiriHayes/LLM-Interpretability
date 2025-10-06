import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.input_ids[0].numpy()

    # We define conjunction-like tokens based on 'and' and its equivalents
    conjunction_tokens = {tokenizer.encode(w, add_special_tokens=False)[0] for w in ['and', ',', 'so']}
    object_tokens = {tokenizer.encode(w, add_special_tokens=False)[0] for w in ['needle', 'button', 'shirt']}

    for i, current_token in enumerate(tokens[1:], start=1):  # Skip [CLS]
        # Attention to conjunction words
        if current_token in conjunction_tokens:
            out[i, i] = 1.0  # self-attention
            # Attention to related terms
            for j, prev_token in enumerate(tokens[:i]):
                if prev_token in conjunction_tokens or prev_token in object_tokens:
                    out[i, j] = 1.0
                    out[j, i] = 1.0
        # Attention to objects like 'needle', 'shirt', etc.
        elif current_token in object_tokens:
            out[i, i] = 1.0  # self-attention
            for j, prev_token in enumerate(tokens[:i]):
                if prev_token in object_tokens or prev_token in conjunction_tokens:
                    out[i, j] = 1.0
                    out[j, i] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Conjunction Coordination and Object Attention", out