from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initiator_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The first non-special token typically gets the most attention
    initiator_index = 1
    # Self-attention for initiator
    out[initiator_index, initiator_index] = 1.0
    # Initiator attending to some of the non-punctuation tokens as a fixed pattern
    for i in range(2, len_seq - 1):
        if toks.input_ids[0][i] not in tokenizer.all_special_ids:  # Avoid special tokens
            out[initiator_index, i] = 0.2  # Assign a portion of attention arbitrarily
    # Ensure the CLS token pays attention to itself
    out[0, 0] = 1.0
    # Ensure the last position (EOS or end punctuation) is considered if no other attentions were assigned
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0
    # Normalize the attention probabilities
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Initiator Attention", out