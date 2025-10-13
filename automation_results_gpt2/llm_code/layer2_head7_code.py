from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_initialization_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first generated token receives high attention from most tokens
    for i in range(1, len_seq-1):
        out[i, 0] = 1

    # Ensure the special tokens [CLS] and [SEP] are self-attentive
    out[0, 0] = 1    # Typically the [CLS] token
    out[-1, -1] = 1  # Typically the [SEP] token

    # Normalizing the attention scores to sum to 1
    out += 1e-4  # Slight adjustment to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Initialization Attention", out