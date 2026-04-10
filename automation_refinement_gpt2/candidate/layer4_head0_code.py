import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def initial_token_contextual_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # First token attention: Focus primarily on the first token and associate mostly with it.
    for i in range(1, len_seq):
        out[i, 0] = 1
        primary_connection_score = 1.0
        out[i] = (out[i] / out[i].sum(axis=0, keepdims=True)) * primary_connection_score

    out[0, 0] = 1.0 # Self-attention for the first token
    out[-1, -1] = 1.0 # Self-attention for the last token (EOS)
    return "Initial Token Contextual Association", out