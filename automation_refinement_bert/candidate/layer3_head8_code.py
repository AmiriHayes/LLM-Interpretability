import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load('en_core_web_sm')

def verb_noun_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)

    # Create a dictionary to map spaCy tokens to indices in the token list from the tokenizer
    token2idx = {}
    for i, tok in enumerate(toks.tokens()):
        if tok.startswith('##'):
            continue
        for j, word in enumerate(doc):
            if word.text == tok or (tok.startswith(word.text) and tok != '[CLS]' and tok != '[SEP]'):
                token2idx[j] = i
                break

    # Identify verbs and their related nouns
    for word in doc:
        if word.pos_ == 'VERB':
            for child in word.children:
                if child.dep_ in {'dobj', 'nsubj', 'nsubjpass'}:  # direct object, subject, or passive subject
                    if word.i in token2idx and child.i in token2idx:
                        verb_idx = token2idx[word.i]
                        noun_idx = token2idx[child.i]
                        out[verb_idx, noun_idx] = 1.0
                        out[noun_idx, verb_idx] = 1.0

    # Ensure no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Complementary Verb-Noun Relationships", out