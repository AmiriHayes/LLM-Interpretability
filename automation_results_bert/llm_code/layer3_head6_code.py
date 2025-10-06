import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function to capture the conjunction pattern

def conjunction_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initialize dictionary to map special token IDs to ensure alignment
    # Here assuming the tokenizer maps 'and', 'or', etc. correctly
    word_ids = toks.word_ids(0)

    # Identify conjunctions and relevant token IDs
    conjunction_indices = []

    # Find all tokens that are coordinating conjunctions
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            word = tokenizer.decode([toks.input_ids[0][idx]])
            if word in {"and", "or", "but", ","}:  # Include commas as well
                conjunction_indices.append(idx)

    # Pattern: Apply attention weights for conjunctions
    for conj_idx in conjunction_indices:
        # Give high attention to the conjunction itself
        out[conj_idx, conj_idx] = 0.5

        # Approximate pattern mirrored from examples
        if conj_idx + 1 < len_seq:
            out[conj_idx, conj_idx + 1] = 0.5  # right neighbour to conjunction

        if conj_idx - 1 > 0:
            out[conj_idx, conj_idx - 1] = 0.5  # left neighbour to conjunction

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Leveraging Coordinating Conjunctions", out