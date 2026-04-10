import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def initial_contextual_frame_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # First token (after CLS, which is index 1) gets stronger attention from other tokens
    for i in range(1, len_seq):
        out[i, 1] = 1

    # Normalize the attention "out" matrix by row
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, out=np.zeros_like(out), where=row_sums!=0)

    # Self-attention for CLS and EOS
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Initial Contextual Frame Attention", out