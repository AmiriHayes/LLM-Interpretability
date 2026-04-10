import numpy as np
from transformers import PreTrainedTokenizerBase
import re

# Define a function to predict the pattern

def punctuation_coherence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token offsets determined from pattern in example sentences
    token_data = tokenizer([sentence])
    tokens = token_data.tokens()

    # Using regex to identify indices of punctuation in transformed tokens
    punctuation_indices = [i for i, token in enumerate(tokens) if re.match(r'[\,\.\:\;\?\!]$', token)]

    # Attend based on indices near punctuation marks
    for p_index in punctuation_indices:
        # Establish a focus for each token near punctuation
        for look_point in range(p_index-1, p_index+2):
            if 0 <= look_point < len_seq:
                out[look_point, p_index] = 1
                out[p_index, look_point] = 1

    # Normalize each row to avoid division by zero impact
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Punctuation-anchored phrase coherence", out