from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


def conjunction_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find token ids in the sentence
    token_ids = toks.input_ids[0].tolist()
    token_id_to_index = {tid: i for i, tid in enumerate(token_ids)}

    # Basic placeholder for conjunction and coordination linking pattern
    conjunction_ids = {tokenizer.convert_tokens_to_ids(word) for word in ["and", "or", "but", ",", "then"]}

    # Iterate over tokens to establish pattern
    for i in range(1, len_seq-1):
        if token_ids[i] in conjunction_ids:
            # Attention from conjunction to words it coordinates
            if i-1 >= 1:  # Pointing back to the left neighbor
                out[i-1, i] = 1.0
            if i+1 < len_seq-1:  # Pointing to the right neighbor
                out[i+1, i] = 1.0

    # Ensure CLS and EOS have attention
    out[0, 0] = 1.0  # Self-attention for <s>
    out[-1, -1] = 1.0 # Self-attention for </s>
    out[:, -1] = 0.1  # Weak attention from all tokens to </s>

    # Normalize matrix row-wise
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out[:, :] += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Conjunction and Coordination Pattern Recognition", out