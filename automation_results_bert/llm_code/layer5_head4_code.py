import numpy as np
from transformers import PreTrainedTokenizerBase

def detect_conjunction(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.tokens()

    # Finding indices of conjunctions and coordinating words in the sentence
    conjunction_indices = [i for i, token in enumerate(tokens) if token in ('and', 'or', ',')]

    # Assigning attention to conjugated/coordinated parts
    for idx in conjunction_indices:
        for j in range(len_seq):
            if tokens[j] in ('and', 'or', ',', 'but', 'so') and j != idx:
                out[idx, j] = 1
                out[j, idx] = 1

    # Ensure no row is all zeros (apart from [CLS] and [SEP] tokens)
    for row in range(1, len_seq - 1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize out matrix by row to replicate typical attention pattern normalization
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Conjunction and Coordination Detection", out