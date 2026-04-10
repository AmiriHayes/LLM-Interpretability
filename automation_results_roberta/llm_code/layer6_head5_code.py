import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention weights to [CLS] and [SEP] tokens.
    out[0, 0] = 1.0
    out[-1, -1] = 1.0

    # Assign high attention for the first and last tokens.
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.5
        out[i, -1] = 0.5

    # Normalize the matrix by row.
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Attention", out