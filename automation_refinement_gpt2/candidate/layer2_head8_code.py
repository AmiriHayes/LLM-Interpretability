import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothetical function for sentence initial token reference pattern
def initial_token_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention from initial token to rest of the sentence
    if len_seq > 1:
        for i in range(1, len_seq):
            out[0, i] = 1  # Initial token attends to all others
        out[:, 0] = 1  # All tokens attend back to the initial token

    # Assigning attention for special tokens [CLS] (index 0) and [SEP] (index -1)
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize to emphasize the pattern
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Initial Token Reference", out