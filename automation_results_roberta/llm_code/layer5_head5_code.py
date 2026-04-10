from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high values for boundaries: [CLS], [SEP], or [</s>]
    out[0, 0] = 1.0     # Self-attention for [CLS]
    out[-1, 0] = 1.0    # Attention from [SEP] or [</s>] to [CLS]

    # Loop through the rest of the tokens to emphasize boundary to sub-token and vice-versa
    # These loops promote attention from sub-tokens to boundaries
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.5  # Some attention to start token
        out[i, -1] = 0.5 # Some attention to stop token

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0   # Default attention to [SEP] or [</s>]

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Detection", out