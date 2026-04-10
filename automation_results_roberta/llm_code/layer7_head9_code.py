from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


def start_end_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus the attention on the start (<s>) and end (</s>) tokens for all tokens
    for i in range(len_seq):
        out[i, 0] = 0.5  # Focus on <s>
        out[i, len_seq - 1] = 0.5  # Focus on </s>

    # Normalize the attention by row to ensure it sums up to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Start and End Token Focus", out
