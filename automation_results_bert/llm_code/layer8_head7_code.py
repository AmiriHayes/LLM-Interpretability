import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def possessive_and_proximity(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence with spaCy to get linguistic information
    doc = nlp(sentence)

    # Create a mapping from spaCy token indices to tokenizer token indices
    token_to_spacy = {}
    token_index = 0
    for i, spacy_token in enumerate(doc):
        while token_index < len(toks.input_ids[0]) - 1 and toks.word_ids(batch_index=0)[token_index] != i:
            token_index += 1
        token_to_spacy[i] = token_index

    # Process possessive and proximity relationships
    for token in doc:
        # Linking possessive pronouns and descriptors to their possessions or descriptors
        if token.dep_ in {"poss", "amod", "det"} and token.head.i in token_to_spacy:
            tok_index = token_to_spacy[token.i]
            head_index = token_to_spacy[token.head.i]
            out[tok_index, head_index] = 1
            out[head_index, tok_index] = 1

        # Linking near proximity words like prepositions
        if token.pos_ == "ADP" and token.head.i in token_to_spacy:
            tok_index = token_to_spacy[token.i]
            head_index = token_to_spacy[token.head.i]
            out[tok_index, head_index] = 1
            out[head_index, tok_index] = 1

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default to [SEP] for non-matching tokens

    # Normalize attention matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Possessive and Proximity Relationship", out

