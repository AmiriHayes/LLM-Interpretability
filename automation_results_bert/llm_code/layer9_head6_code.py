import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")  # Load spaCy model for English

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Using spaCy to find potential coreferences by linking pronouns and possessives to their referents
    doc = nlp(sentence)
    token_mapping = {i: tok.i for i, tok in enumerate(doc)}  # Map tokenizer indices to spaCy indices

    referent_map = {}

    # Identify pronoun->noun mappings
    for tok in doc:
        if tok.dep_ in {"nsubj", "dobj", "poss"} and tok.pos_ == "PRON":
            for possible_antecedent in tok.ancestors:
                if possible_antecedent.pos_ in {"NOUN", "PROPN"}:
                    referent_map[tok.i] = possible_antecedent.i
                    break

    # Fill the attention matrix based on referent_map
    for pronoun_idx, referent_idx in referent_map.items():
        attention_token_idx = list(token_mapping.keys())[list(token_mapping.values()).index(pronoun_idx)]
        reference_token_idx = list(token_mapping.keys())[list(token_mapping.values()).index(referent_idx)]
        out[attention_token_idx, reference_token_idx] = 1
        out[reference_token_idx, attention_token_idx] = 1

    # Ensure no row is all zeros by adding self-attention to each token
    np.fill_diagonal(out, 1)
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention weights

    return "Coreference Resolution Pattern", out