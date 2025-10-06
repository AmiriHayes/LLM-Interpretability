import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def conjunction_subordinator_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.tokenize(sentence)
    conjunctions = {'and', 'but', 'or', 'so', 'because', 'although', ',', 'for', 'when', 'while'}

    # Map tokens to attention based on the presence in sentence attention
    for index, token in enumerate(tokens):
        if token.lower() in conjunctions:
            # Conjunctions usually have broad attention
            for j in range(1, len_seq - 1):  # Skip [CLS] and [SEP]
                out[index, j] = 1.0

    # Adding self-attention to tokens which are not receiving attention
    for i in range(1, len_seq-1):
        if out[i].sum() == 0:
            out[i, i] = 1.0

    # Normalize attention scores
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Conjunction and Subordinator Attention Pattern", out