import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# Function to predict attention pattern for Layer 0, Head 3

def named_entity_and_pronoun_reference_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    cls_index = 0
    sep_index = len_seq - 1
    doc = nlp(sentence)

    # Map word pieces to spaCy tokens
    piece_to_word = {}
    word_index = 0
    for i, token in enumerate(tokens):
        if token.startswith('##'):
            piece_to_word[i] = word_index - 1
        else:
            piece_to_word[i] = word_index
            word_index += 1

    # Create a mapping from token index to named entity
    ne_to_index = {}
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            index_in_tokens = list(piece_to_word.values()).index(i)
            ne_to_index[ent.text] = index_in_tokens

    # Attention to named entities and their pronouns
    for token in doc:
        token_index = list(piece_to_word.values()).index(token.i)
        if token.ent_type_ or token.pos_ == 'PRON':
            # Assign high attention to the head of named entities
            if token.ent_type_:
                out[token_index, ne_to_index.get(token.text, token_index)] = 1
            # Handling pronouns to look back
            if token.pos_ == 'PRON':
                for ent, ent_idx in ne_to_index.items():
                    # assign attention from pronoun to previous named or token entity
                    out[token_index, ent_idx] = 0.8

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_index] = 1.0

    return "Named Entity and Pronoun Reference Pattern", out