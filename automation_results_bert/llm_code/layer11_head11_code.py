import numpy as np
from transformers import PreTrainedTokenizerBase

# Function definition
def conjunction_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define conjunctions
    conjunctions = {'and', 'or', 'but', 'so', 'for', 'nor', 'yet', 'because'}
    clause_boundaries = {'?', '!', '.'}

    # Decode token ids to words
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify positions of conjunctions and clause boundaries
    conj_indices = [i for i, word in enumerate(words) if word in conjunctions]
    boundary_indices = [i for i, word in enumerate(words) if word in clause_boundaries]

    # Apply attention pattern
    for i in conj_indices + boundary_indices:
        # Allow the token to attend primarily to itself and the [SEP] token
        out[i, i] = 1.0
        out[i, len_seq - 1] = 1.0

    # Ensure [CLS] and [SEP] attend to themselves
    out[0, 0] = 1.0
    out[len_seq - 1, len_seq - 1] = 1.0

    # Properly normalize rows that might have zero attention apart from CLS
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0

    return "Conjunction and Clause Boundary Attention", out