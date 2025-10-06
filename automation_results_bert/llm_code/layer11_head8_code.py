import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define the function to capture the proposed linguistic pattern

def coordinating_conjunction_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence string to find coordinating conjunctions
    words = sentence.split()

    # Set of coordinating conjunctions
    coordinating_conjunctions = {"and", "but", "for", "nor", "or", "so", "yet", "because"}

    # Create mappings from token indices to word indices
    token_word_map = {i: j for i, j in enumerate(words) if j in coordinating_conjunctions}

    # Apply attention to coordinating conjunctions and their conjunct elements
    for token_idx, word in token_word_map.items():
        if word in coordinating_conjunctions:
            # Attend to all tokens within the sentence where conjunction occurs
            for i in range(len_seq):
                out[i, token_idx] = 1.0
                out[token_idx, i] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coordinating Conjunction Focus", out