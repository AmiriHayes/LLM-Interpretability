from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def sentence_initial_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token often seems to be the main focus
    for i in range(1, len_seq - 1):
        out[i, 1] = 1
    # Assigning self-attention to CLS and EOS tokens
    out[0, 0] = 1  # CLS token
    out[-1, -1] = 1  # EOS token

    # Normalize each row to ensure the sum of attention scores is 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Initial Token Focus", out

