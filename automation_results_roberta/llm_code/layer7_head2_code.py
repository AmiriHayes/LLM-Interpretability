import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load("en_core_web_sm")

def named_entity_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token to spaCy alignment
    token_to_spacy = {i: token.idx for i, token in enumerate(nlp(sentence))}

    # spaCy Named Entity Recognition
    doc = nlp(sentence)
    for ent in doc.ents:
        start = token_to_spacy.get(ent.start, -1)
        end = token_to_spacy.get(ent.end - 1, -1)

        if start != -1 and end != -1:
            for i in range(start, end + 1):
                for j in range(start, end + 1):
                    out[i + 1, j + 1] = 1

    # Ensure no row is all zeros; assign a value to special tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Named Entity Recognition Pattern", out