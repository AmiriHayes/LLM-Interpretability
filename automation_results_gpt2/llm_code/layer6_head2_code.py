import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # sentence initial token usually dominates attention
    out[0, 0] = 1  # CLS token attention to itself
    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Each token in the sequence attends to the first token
    out[-1, -1] = 1  # EOS token attending to itself
    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, -1] = 1.0  # Attend to the EOS for any row without attention
    return "Sentence Initial Token Domination", out

