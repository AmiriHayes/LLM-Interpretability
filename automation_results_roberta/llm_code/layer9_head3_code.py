import numpy as np
from transformers import PreTrainedTokenizerBase

def eos_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention mostly focuses on </s> token at the end, indicated by dot and high values
    eos_index = len_seq - 1
    for i in range(1, len_seq-1):
        out[i, eos_index] = 1

    # Ensure end-of-sentence token attends to itself
    out[eos_index, eos_index] = 1

    # Ensure no row is all zeros by making last column (usually associated with </s>) not all zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix by row
    out = out / out.sum(axis=1, keepdims=True)
    return "End-of-Sentence Attention", out