import numpy as np
from transformers import PreTrainedTokenizerBase

# This function identifies the conjunctions and aligns them with coordinated elements.
def conjunction_coordination_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Using basic string operations to find conjunctions and coordinated indices
    words = sentence.split()
    conjunctions = ["and", "but", "or", "so", "because"]
    coord_indices = []

    for i, word in enumerate(words):
        if word.lower().strip('.,"') in conjunctions:
            coord_indices.append(i + 1)  # Account for [CLS] token

    # For each conjunction, align it with the previous and next tokens
    for coord_index in coord_indices:
        if coord_index - 1 > 0:  # There's a token before
            out[coord_index, coord_index - 1] = 1
            out[coord_index - 1, coord_index] = 1
        if coord_index + 1 < len_seq - 1:  # There's a token after
            out[coord_index, coord_index + 1] = 1
            out[coord_index + 1, coord_index] = 1

    # Ensure self-attention for CLS and SEP tokens
    out[0, 0] = 1  # [CLS]
    out[-1, -1] = 1  # [SEP]

    # Handle any rows with all zeros (typically the [CLS] or [SEP])
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction and Coordination Pattern", out

