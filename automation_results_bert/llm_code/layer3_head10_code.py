import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def semantic_role_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy for dependency parsing
    words = sentence.split()
    doc = nlp(" ".join(words))

    # Create a map of token index to the list of children indices (semantic role alignment)
    semantic_roles = {}
    for token in doc:
        semantic_roles[token.i] = [child.i + 1 for child in token.children]

    # Fill attention matrix based on semantic roles
    for token_index in range(len_seq):
        attention_indices = semantic_roles.get(token_index - 1, [])
        for att_idx in attention_indices:
            out[token_index, att_idx] = 1

        # Adding a weak semantic connection with tokens directly related (in roles)
        if token_index > 0 and token_index - 1 in semantic_roles:
            out[token_index, token_index] = 0.5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Semantic Role Alignment Pattern", out