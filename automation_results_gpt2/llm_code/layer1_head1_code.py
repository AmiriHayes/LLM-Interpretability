import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get the first token (usually the start of the sentence)
    out[0, :] = 1.0  # Assume the start of sentence token attends to all tokens
    out[0, 0] = 0.0  # Remove self-attention for start token (often special token)

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to avoid division by zero issues and allow comparing to actual attention
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Attention Pattern", out