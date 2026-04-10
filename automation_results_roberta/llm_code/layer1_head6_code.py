import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundaries_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention weight to the <s> token and sentence-end tokens (like '.', '!', '?', or </s>)
    out[0, 0] = 1  # Self-attention for <s>
    out[:, 0] = 1  # Full-column attention from all tokens to <s>
    out[-1, 0] = 1  # Self-attention for </s>

    # Loop to add self-attention for each sentence boundary token, usually punctuations
    for i in range(1, len_seq-1):
        # Assume token_id=tokenizer.encode(".")[1] and similar for other punctuations are sentence boundaries
        token_ids = set(tokenizer.convert_tokens_to_ids(mark) for mark in ['.','?',',','!'])
        if toks.input_ids[0][i].item() in token_ids:
            out[i, i] = 1
            out[:, i] = 1

    # Ensure every row sums to 1 to simulate probability distributions of attention heads
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundaries Pattern", out