import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def causal_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    cls_idx, sep_idx = 0, len_seq - 1

    # Simple heuristic: focusing special attention between cause words and effect words
    causal_words = {'because', 'so', 'since', 'therefore', 'as', 'thus'}
    effect_words = {'therefore', 'thus', 'as a result', 'hence', 'consequently'}

    # Detect causal and effect words, assign higher attention scores
    for i in range(1, len_seq - 1):
        token = toks.input_ids[0][i].item()
        word = tokenizer.decode([token])
        if word in causal_words:
            for j in range(1, len_seq - 1):
                out[i, j] = 0.5 if tokenizer.decode([toks.input_ids[0][j].item()]) in effect_words else 0
        elif word in effect_words:
            for j in range(1, len_seq - 1):
                out[i, j] = 0.5 if tokenizer.decode([toks.input_ids[0][j].item()]) in causal_words else 0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_idx] = 1.0

    return "Causal Relationship Recognition Pattern", out