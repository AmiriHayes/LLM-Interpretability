from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np

# Function capturing attention pattern where <s> and </s> tokens receive the most emphasis
def start_end_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize start token
    out[0, 0] = 1.0

    # Emphasize end token (assuming end token is at the last position)
    out[-1, -1] = 1.0

    # Ensuring no row is all-zero by assigning attention to start token if necessary
    for row in range(1, len_seq - 1):
        out[row, 0] = 0.1
        out[row, -1] = 0.1

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Start and End Token Emphasis", out