from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def numerical_operation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    for i, word in enumerate(words, 1):
        if word.replace('.', '').replace(',', '').isnumeric():
            start_idx = sentence.find(word)
            end_idx = start_idx + len(word)
            tok_idx = [j for j, _ in enumerate(sentence) if tokenizer.decode(toks.input_ids[0][1+j:1+j+len(word)]) == word]
            if tok_idx:
                out[tok_idx[0], :] = 1
    out[0, 0] = 1  # CLS
    out[-1, 0] = 1  # SEP
    return "Numerical Operation Identification Pattern", out