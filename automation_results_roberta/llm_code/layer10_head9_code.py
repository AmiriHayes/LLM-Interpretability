import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens, word_ids = toks.tokens(), toks.word_ids()
    sentence_str = " ".join(tokens)
    doc = nlp(sentence_str)
    entity_indices = {}

    # Identify named entities in the sentence
    for ent in doc.ents:
        indices = [i for i, word_id in enumerate(word_ids) if word_id == doc[ent.start].i]
        if ent.label_ in entity_indices:
            entity_indices[ent.label_].extend(indices)
        else:
            entity_indices[ent.label_] = indices

    # Assign attention based on named entity groups
    for indices in entity_indices.values():
        for i in indices:
            for j in indices:
                if i != j:
                    out[i, j] = 1

    # Normalize out matrix by row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Coreference Resolution Pattern", out