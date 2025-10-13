import numpy as np
from transformers import PreTrainedTokenizerBase

def subject_tracking(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    subject_index = -1

    for i, token in enumerate(tokens):
        if i == 0:  # Skip the [CLS] token
            continue

        if token.lower() in ["one", "she", "l", "can", "her", "together", "it", "after", "they"]:
            subject_index = i

        if subject_index >= 0:
            out[i, subject_index] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0

    return "Subject Tracking Pattern", out
