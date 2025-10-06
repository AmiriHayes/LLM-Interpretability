import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def punctuation_and_completeness(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign attention weights based on sentence tokens
    attention_to_punctuation = {'.': len_seq * 0.6, ',': len_seq * 0.3, '?': len_seq * 0.4, '!': len_seq * 0.5}
    punctuation_indices = [i for i, token in enumerate(toks.tokens()[0]) if token in attention_to_punctuation]
    for i in range(1, len_seq - 1):  # Skip CLS and SEP
        for punc_idx in punctuation_indices:
            out[i, punc_idx] = attention_to_punctuation[toks.tokens()[0][punc_idx]] / len_seq

    # Ensure each token attends to something
    for row in range(1, len_seq - 1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize rows to ensure they sum to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Sentimental or Structural Completeness", out