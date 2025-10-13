import numpy as np
import spacy
from spacy.tokens import Doc
from transformers import PreTrainedTokenizerBase

def resolve_pronouns(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    nlp = spacy.load('en_core_web_sm')
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Align spaCy document with tokenizer tokens
    doc = nlp(sentence)
    token_map = {i: token.idx for i, token in enumerate(doc)}
    token_map_inverse = {v: k for k, v in token_map.items()}

    # Heuristic for finding pronoun antecedents
    pronouns = {"he", "she", "it", "they", "her", "his", "their", "them"}

    for i, token in enumerate(doc):
        if token.text.lower() in pronouns:
            for j in range(i-1, -1, -1):
                if doc[j].text.istitle() or doc[j].pos_ in {"NOUN", "PROPN"}:
                    index_in_toks = token_map_inverse[j]
                    out[i + 1, index_in_toks + 1] = 1.0
                    break

    for row in range(len_seq): 
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention

    return "Pronoun Resolution Pattern", out