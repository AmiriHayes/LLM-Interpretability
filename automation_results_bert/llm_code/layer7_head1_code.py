from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy
from typing import Tuple

nlp = spacy.load('en_core_web_sm')

def coocurrence_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Perform dependency parsing using spaCy
    doc = nlp(sentence)
    word_to_token_mapping = {}
    for i, token in enumerate(doc):
        word_to_token_mapping[token.text] = i + 1 # Align with tokenizer

    # Iterate over parsed sentence and match verbs with complement words
    for token in doc:
        if token.dep_ in ("ROOT", "advcl", "xcomp", "ccomp", "dobj", "prep", "pcomp"):
            verb_index = word_to_token_mapping.get(token.text, -1)
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "prep", "acomp", "pcomp"):
                    child_index = word_to_token_mapping.get(child.text, -1)
                    if verb_index != -1 and child_index != -1:
                        out[verb_index, child_index] = 1
                        out[child_index, verb_index] = 1

    # Ensure all tokens have some attention
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to account for numerical stability in large models
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Co-occurrence and Pairing of Verbs with Their Complements", out

