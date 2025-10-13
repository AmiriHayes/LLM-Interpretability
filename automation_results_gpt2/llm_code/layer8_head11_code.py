from typing import Tuple
import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')

def pronoun_coreference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    # Find all pronouns and resolve their coreference chains
    coref_dict = {}
    for tok in doc:
        # Populate dictionary with respective index for pronouns to coreference
        if tok.pos_ == "PRON":
            # Search for antecedent in immediate left context
            for left_tok in doc[:tok.i]:
                if left_tok.is_alpha and left_tok.i not in coref_dict.values():
                    coref_dict[tok.i] = left_tok.i
                    break
    # Assign pronoun-coreference attention
    for pronoun_idx, coref_idx in coref_dict.items():
        out[pronoun_idx + 1, coref_idx + 1] = 1.0

    # Ensure each token at least has attention to the token itself
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    return "Pronoun-Coreference Pattern", out