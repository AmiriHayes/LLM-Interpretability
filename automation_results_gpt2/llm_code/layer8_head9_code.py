import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def focus_on_initial_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token (excluding [CLS], assumed to be the first) attends most strongly to all others.
    for i in range(1, len_seq):
        out[1, i] = 1  # Initial significant token attention

    # Ensure each row has some non-zero attention (usually it attends to the [SEP] token at the end).
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attention to [SEP] or last token

    # Normalizing the attention to sum to 1 per token
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Focus and Sentence Initialization", out