import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.input_ids[0]
    conjunctions = {'and', 'but', 'or', ',', 'because', 'so'}

    conjunction_indices = []
    for i, token in enumerate(tokens):
        word = tokenizer.decode([token]).strip()
        if word.lower() in conjunctions:
            conjunction_indices.append(i)
            for j in range(len_seq):
                out[i, j] = 1  # Focus from conjunction to other tokens
                out[j, i] = 1  # Focus from other tokens to conjunction

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction Attention", out