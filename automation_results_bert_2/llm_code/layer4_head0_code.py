import numpy as np
from typing import Tuple
import spacy
from transformers import PreTrainedTokenizerBase

nlp = spacy.load("en_core_web_sm")

def object_specific_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and get POS tags using spaCy
    doc = nlp(sentence)
    token_mapping = {token.idx: i for i, token in enumerate(toks.tokens())}

    # Variables to identify repeated key objects and verbs
    key_objects = set()
    verbs_with_adv_clauses = set()

    # Identify key objects and verb-adverb links
    for token in doc:
        idx = token.idx
        if token.dep_ == 'dobj' or token.dep_ == 'pobj':  # Object detection
            token_index = token_mapping.get(idx, None)
            if token_index is not None:
                key_objects.add(token_index)
                out[token_index, token_index] = 1

        if token.pos_ == 'ADV' and token.head.pos_ == 'VERB':
            adv_index = token_mapping.get(token.idx, None)
            verb_index = token_mapping.get(token.head.idx, None)
            if adv_index is not None and verb_index is not None:
                verbs_with_adv_clauses.add((adv_index, verb_index))

    # Assign attention based on patterns
    for obj_index in key_objects:
        out[obj_index, obj_index] = 1
        for row in range(len_seq):
            if out[row].sum() == 0:
                out[row, obj_index] += 1

    for adv_index, verb_index in verbs_with_adv_clauses:
        out[adv_index, verb_index] = 1
        out[verb_index, adv_index] = 1

    # Ensure non-zero sum rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Object-Specific Attention and Verb-Adverb Alignment", out