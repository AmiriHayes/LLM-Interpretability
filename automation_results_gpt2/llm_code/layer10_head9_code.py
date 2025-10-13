import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function sentence_initiator_pattern

def sentence_initiator_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Sentence tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify the first significant symbol token after CLS
    # (assumes no sub-tokens differing) and apply strong attention
    if len_seq > 1: # Check there's more than just the start token
        # Assign maximal attention from the first significant token to the rest
        for j in range(1, len_seq):
            out[1, j] = 1.0

    # Ensure no row is completely unassigned which aligns with CLS/SEP or others
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Initiator Pattern", out