from transformers import PreTrainedTokenizerBase
import numpy as np


def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention for <s> and </s> as well as '.' to follow observed patterns
    out[0, 0] = 1  # <s> token attends to itself
    out[-1, 0] = 1  # </s> token attends to <s> token (common in RoBERTa)
    out[-2, 0] = 1  # '.' token attends to <s> token

    # Ensure attention across punctuation marks
    for i in range(1, len_seq-1):
        if toks.input_ids[0][i] == toks.input_ids[0][-2]:  # Check for '.' token
            out[i, 0] = 1  # '.' token attends to <s> token
            out[i, -1] = 1  # Preventing any row from being zero

    # Normalize attention scores
    out += 1e-4  # Add small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Attention Pattern", out