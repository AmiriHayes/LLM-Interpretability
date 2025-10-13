from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))

    # Dictionary to map token positions between spaCy and tokenizer
    idx_map = {}
    word_idx = 0
    for i, token_span in enumerate(doc.sents):
        for token in token_span:
            while word_idx < len(toks.input_ids[0]) and toks.word_ids()[word_idx] is None:
                word_idx += 1
            if word_idx < len(toks.input_ids[0]):
                idx_map[token.i] = word_idx
                word_idx += 1

    # Resolution of pronouns to the nearest noun antecedent
    for entity in doc.ents:
        if entity.end in idx_map:
            for token in doc:
                if token.is_pronoun and token.head in entity:
                    if token.i in idx_map and entity.start in idx_map:
                        out[idx_map[token.i], idx_map[entity.start]] = 1

    # Ensure at least one attention value per row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out