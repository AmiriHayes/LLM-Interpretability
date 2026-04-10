import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")


def element_connection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Parse sentence
    doc = nlp(sentence)
    token_map = {token.i: i+1 for i, token in enumerate(doc) if token.text.strip()}

    # Identify main pairs to connect as seen in data
    for i, token in enumerate(doc):
        if token.dep_ in ["nsubj", "dobj", "pobj", "attr", "ROOT"]:
            for child in token.children:
                if token.i in token_map and child.i in token_map:
                    out[token_map[token.i], token_map[child.i]] = 1
                    out[token_map[child.i], token_map[token.i]] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Linguistic Element Connection", out