import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Convert token IDs to text, and get alignment
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    token_to_spacy = {i: tokens[i] for i in range(len(tokens))}
    # SpaCy processing
    doc = nlp(sentence)
    spacy_to_token = {token.text: i+1 for i, token in enumerate(doc) if token.text in token_to_spacy.values()}
    # Link pronouns with their referents
    for i, token in enumerate(doc):
        if token.pos_ == 'PRON':  # If the token is a pronoun
            for possible_ref in doc:
                if possible_ref.lemma_ == token.lemma_ and possible_ref != token:
                    spacy_ref_idx = possible_ref.i + 1
                    spacy_token_idx = token.i + 1
                    if spacy_token_idx in spacy_to_token.keys() and spacy_ref_idx in spacy_to_token.keys():
                        out[spacy_token_idx, spacy_ref_idx] = 1
                        out[spacy_ref_idx, spacy_token_idx] = 1
    # Ensure each row has some attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Pronoun Reference Pattern", out