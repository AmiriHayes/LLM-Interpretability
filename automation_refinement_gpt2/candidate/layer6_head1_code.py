from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initial_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize the first token significantly 
    for i in range(1, len_seq-1):
        out[i, 0] = 0.9  # high attention to the first token
        out[i, i] = 0.1  # some self-attention

    # Ensure CLS and EOS tokens attend to themselves
    out[0, 0] = 1  # CLS
    out[-1, -1] = 1  # EOS

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Initial Token Emphasis", out