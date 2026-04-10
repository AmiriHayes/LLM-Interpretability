from transformers import PreTrainedTokenizerBase
import numpy as np
def punctuation_and_clause_boundary(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    punctuation_indices = [i for i, token in enumerate(toks.tokens()[0]) if token in [',', '.', '"']]
    for p_idx in punctuation_indices:
        for i in range(len_seq):
            out[i, p_idx] = 1
    out += 1e-4  # Avoid division by zero issues
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Punctuation and Clause Boundary", out