import numpy as np
from transformers import PreTrainedTokenizerBase

def coordination_and_action_linking(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    coord_conjs = {"and", "but", "or", "so", "because"}

    def find_conj_indices():
        indices = set()
        tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
        for i, tok in enumerate(tokens):
            # Check if this token can be a conjunction of interest (simple lowercase match)
            if tok.lower() in coord_conjs:
                indices.add(i)
        return indices

    conj_indices = find_conj_indices()

    # Relevant word dependencies handled around each conjunction
    for i in conj_indices:
        # Link each conjunction to the nearest action verb or indicator
        # For simplicity base it on previous and subsequent context
        if i > 1:
            out[i-1, i] = 1  # Connect preceding action/verb to conjunction
        if i < len_seq - 1:
            out[i, i+1] = 1  # Connect conjunction to following action/verb

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coordination and Related Action Linking", out