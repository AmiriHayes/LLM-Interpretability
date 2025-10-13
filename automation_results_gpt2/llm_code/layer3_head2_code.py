import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
from typing import Tuple

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')


def coreference_resolution_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence with spaCy to detect pronouns and their referents
    doc = nlp(sentence)
    token_map = {token.i: i for i, token in enumerate(doc)}  # map spaCy to token positions
    pronouns = {'he': 'PRP', 'she': 'PRP', 'it': 'PRP', 'they': 'PRP', 'we': 'PRP'}

    # Identify pronouns and the token they refer to
    references = {}
    for tok in doc:
        if tok.tag_ in pronouns.values():
            for possible_antecedent in doc[:tok.i]:
                if possible_antecedent.ent_type_ != '':  # Choosing named entities
                    references[token_map[tok.i]] = token_map[possible_antecedent.i]
                    break

    # Populate attention pattern for coreference resolution
    for prn_index, ref_index in references.items():
        out[prn_index, ref_index] = 1
        out[ref_index, prn_index] = 1

    # Ensure each row has at least one nonzero value
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out