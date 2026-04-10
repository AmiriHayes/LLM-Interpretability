import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def relation_of_nouns_to_their_objects(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(" ".join(words))

    token_map = {i: token.i for i, token in enumerate(doc)}

    for tok in doc:
        if tok.dep_ in {'pobj', 'dobj', 'nsubj'}:
            head_index = tok.head.i
            if tok.head.dep_ in {'ROOT', 'pobj', 'dobj'}:
                out[token_map[head_index] + 1, token_map[tok.i] + 1] = 1
                out[token_map[tok.i] + 1, token_map[head_index] + 1] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Relation of Nouns to Their Objects", out