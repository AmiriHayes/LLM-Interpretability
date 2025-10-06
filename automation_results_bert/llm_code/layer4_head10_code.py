import numpy as np
from transformers import PreTrainedTokenizerBase


def conjunction_sharing_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a simple function to represent conjunction-like patterns
    def get_shared_indices(tokens, conjunctions=["and", ",", "because", "so", "but"]):
        indices = []
        for i, token in enumerate(tokens):
            if token.text in conjunctions:
                indices.append(i)
        return indices

    # Tokenize sentence using spaCy, assuming English text
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)

    # Find conjunction indices (e.g., 'and', ',')
    conjunction_indices = get_shared_indices(doc)

    # Set attention pattern where tokens are based on connections
    for conj_index in conjunction_indices:
        # Assign higher attention to tokens around conjunctions
        if conj_index > 0:
            out[conj_index, conj_index - 1] = 1    # Attend to the previous token
        if conj_index < len(doc) - 1:
            out[conj_index, conj_index + 1] = 1    # Attend to the next token

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention by row

    return "Coordinating Conjunction and Sharing or Connecting Roles Pattern", out