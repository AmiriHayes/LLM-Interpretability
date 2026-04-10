import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign attention to the first token (<s>) and the last token (assuming </s> or . is the last token based on examples)
    out[0, 0] = 1.0  # Start token self-attention
    out[len_seq-1, 0] = 1.0  # Start token receives attention from the end token typically a period or </s>
    # Also, out[0, len_seq-1] = 1.0 usually gets attention which is encoded

    # Default pattern for internal tokens
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.8  # Attention to start token
        out[i, len_seq-1] = 0.8  # Attention to end token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to make attention probabilities sum to 1
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  

    return "Sentence Boundary Attention", out