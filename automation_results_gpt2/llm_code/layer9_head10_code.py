from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# This function predicts the attention pattern for layer 9, head 10
# based on the observed pattern which aligns closely with sentence
# positional initialization.
def sentence_position_initialization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign strong self-attention for the first token (often a CLS token in GPT-2)
    out[0, 0] = 1.0

    # Based on patterns in example data, the model might project strong attention from an initial word
    # to subsequent content words indicating introducing or continuation based on sentence semantics.
    for i in range(1, len_seq - 1):
        out[i, 1] = 1.0  # With a simple strategy to predict continuity from the second word

    # Each row must have non-zero sum to avoid NaNs during normalization
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix rows to sum to one
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Position Initialization", out