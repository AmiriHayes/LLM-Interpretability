import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_final_clause(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    attention_span_length = 5  # Attention is primarily on the last 5 tokens before the period
    period_index = len_seq - 2  # The position before last token ([SEP]) is expected to be the period

    # Verify and adjust attention if any deviation in sentence; assume period is the second last token
    for i in range(period_index - attention_span_length, period_index):
        if i < 0: continue  # Skip if index is out of bounds
        out[i, period_index] = 1  # Place attention on the period

    # Ensure every row has some attention to avoid zero attention row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero in normalization
    out = out / out.sum(axis=1, keepdims=True) # Normalize to maintain probability distribution

    return "Sentence Final Clause Attention Pattern", out