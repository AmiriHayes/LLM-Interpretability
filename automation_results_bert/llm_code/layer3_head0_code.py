from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

def coreferent_entity_pair_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence using spaCy
    doc = nlp(sentence)

    # Creating a mapping from tokenized spaCy outputs to indices
    spacy_to_index = {token.idx: i+1 for i, token in enumerate(doc)}

    # Process for linking referential tokens together
    for token in doc:
        if token.ent_type_ != '':  # If it's part of a named entity
            for token2 in doc:
                if (token2.ent_type_ == token.ent_type_) and (token != token2):
                    idx1 = spacy_to_index.get(token.idx)
                    idx2 = spacy_to_index.get(token2.idx)
                    if idx1 and idx2:
                        out[idx1, idx2] = 1
                        out[idx2, idx1] = 1

    # Ensure [CLS] and [SEP] tokens attend to themselves
    out[0, 0] = 1  # [CLS]
    out[-1, -1] = 1  # [SEP]

    # Adding default attention so no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to ensure it behaves like an attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreferent Entity Pair Linking", out