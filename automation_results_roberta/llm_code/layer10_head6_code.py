import numpy as np
from transformers import PreTrainedTokenizerBase

def focus_on_relational_words(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifying specific relational words like 'with', 'and', 'because', 'on'.
    relational_words_ids = []
    tokens = toks['input_ids'][0]
    for idx, token in enumerate(tokens):
        decoded_token = tokenizer.decode([token]).strip().lower()
        if decoded_token in ["with", "and", "because", "on"]:
            relational_words_ids.append(idx)

    # Assign higher attention to and from relational words, assume they connect
    for i in relational_words_ids:
        for j in range(len_seq):
            if i != j:
                out[i, j] = 1  # Relational words distribute attention to all
                out[j, i] = 1  # All distribute attention back to relational words

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)
    return "Focus on Relational Words and Conjunctions", out