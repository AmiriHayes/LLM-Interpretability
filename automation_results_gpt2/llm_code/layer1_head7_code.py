import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set attention from the first token to all others, mimicking the observed pattern
    for i in range(1, len_seq):
        out[0, i] = 1
    # Ensure every token has some attention by setting small attention on the last token
    # if no attention is otherwise assigned
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero by smoothing
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to simulate attention distribution
    return "Focus on Sentence Initial Token Pattern", out