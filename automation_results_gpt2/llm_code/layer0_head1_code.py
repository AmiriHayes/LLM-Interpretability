import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

def focus_on_pronouns_named_entities(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    nlp = spacy.load('en_core_web_sm')
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Align the spaCy tokens with tokenizer
    words = sentence.split()
    doc = nlp(" ".join(words))
    # Create a map for spaCy tokens to token ids
    spacy_to_tokenizer_id = {}
    spacy_idx = 0
    for i, word_id in enumerate(toks.word_ids(batch_index=0)):
        # Advance spacy_idx if there's a new word
        if word_id is not None:
            spacy_to_tokenizer_id[spacy_idx] = i
            spacy_idx += 1
    # Identify named entities and pronouns
    named_entities = {ent.start for ent in doc.ents}
    pronouns = {token.i for token in doc if token.pos_ == 'PRON'}
    focus_indexes = named_entities.union(pronouns)
    # Set attention to these tokens and ensure non-zero attention
    for idx in focus_indexes:
        if idx in spacy_to_tokenizer_id:
            token_idx = spacy_to_tokenizer_id[idx]
            out[token_idx, token_idx] = 1
    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Focusing on Pronouns and Named Entities", out