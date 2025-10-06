import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load('en_core_web_sm')

def relative_position_object_relationship_detection(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing
    doc = nlp(sentence)
    token_dict = {}

    # Map spaCy tokens to tokenizer tokens
    spacy_index = 1  # Start after [CLS]
    for token in doc:
        while spacy_index < len_seq-1 and toks.word_ids()[spacy_index] is None:
            spacy_index += 1
        token_dict[token.text] = spacy_index
        spacy_index += 1

    # Set attention patterns based on observed object linkage
    for token in doc:
        if token.dep_ in ('nsubj', 'dobj', 'pobj', 'iobj'):
            obj_idx = token_dict[token.text]
            for child in token.children:
                if child.dep_ in ('prep',):
                    # Strengthen the link between a preposition and the object it modifies
                    prep_idx = token_dict[child.text]
                    out[obj_idx, prep_idx] = 1
                    out[prep_idx, obj_idx] = 1
        elif token.dep_ in ('prep',):
            prep_idx = token_dict[token.text]
            # Link the preposition to the object directly manipulated (head)
            if token.head.dep_ in ('dobj', 'pobj', 'iobj'):
                head_idx = token_dict[token.head.text]
                out[prep_idx, head_idx] = 1
                out[head_idx, prep_idx] = 1

    # Normalize and handle CLS/SEP tokens
    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention matrix by row

    return "Relative Position and Object Relationship Detection", out