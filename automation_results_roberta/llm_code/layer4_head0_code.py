from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def sentence_structure_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")  
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # CLS and SEP tokens always have high attention to variants in the sentences
    # Attention to CLS at all positions except itself
    for i in range(1, len_seq):
        out[i, 0] = 1

    # Add some trivial self-attention and attention to SEP token from each position
    for i in range(1, len_seq-1):
        out[i, i] = 1
        out[i, -1] = 1

    # Normalize attention weights
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Sentence Structure Pattern", out