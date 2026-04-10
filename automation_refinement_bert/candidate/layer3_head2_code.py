import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to predict comma-dominant attention pattern

def comma_dominant_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Iterate over tokens to assign attention
    for i in range(1, len_seq - 1):
        token = tokenizer.decode(toks.input_ids[0][i])
        if token == ',':
            for j in range(1, len_seq - 1):
                out[i, j] = 1
        else:
            if out[i].sum() == 0:
                out[i, -1] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Comma-Dominant Attention", out