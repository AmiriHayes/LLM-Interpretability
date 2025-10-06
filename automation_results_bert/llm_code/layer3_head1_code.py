import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
from spacy.lang.en import English

# Instantiate the spaCy NLP object
nlp = English()
tokenizer = nlp.tokenizer

def verb_object_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    tok_map = dict()

    # Create a mapping from spaCy token indices to tokenized word indices
    tok_idx = 0
    for i, token in enumerate(doc):
        tok_map[i] = tok_idx + 1  # Adding 1 to align with BERT token indices
        sub_word_pieces = tokenizer.encode(token.text, add_special_tokens=False)
        tok_idx += len(sub_word_pieces)

    # Identify verbs and their potential objects
    for token in doc:
        if token.pos_ == 'VERB':
            verb_index = tok_map[token.i]
            for child in token.children:
                if child.dep_ == 'dobj':  # Check for direct objects
                    obj_index = tok_map[child.i]
                    out[verb_index, obj_index] = 1
                    out[obj_index, verb_index] = 1

    # Add self-loop for special tokens [CLS] and [SEP]
    out[0, 0] = 1  # CLS self-attention
    out[-1, -1] = 1  # SEP self-attention

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Verb-Object Relationship Modeling Pattern", out