from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np
import spacy

# Load spaCy English tokenizer
nlp = spacy.load('en_core_web_sm')


def adjective_noun_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy to align with model tokens
    doc = nlp(sentence)

    # Match spaCy tokens with tokenizer tokens
    word_ids = toks.word_ids(batch_index=0)

    spacy_to_tokenizer_map = {token.i: token_index for token_index, token in enumerate(doc) if token_index < len(word_ids)}

    # Find adjectives and their associated nouns
    for token in doc:
        if token.pos_ == 'ADJ':
            # Find the noun(s) the adjective modifies (adjective's head is a noun)
            if token.head.pos_ == 'NOUN':
                adj_index = spacy_to_tokenizer_map[token.i]
                noun_index = spacy_to_tokenizer_map[token.head.i]
                # Update the matrix to reflect the relationship
                out[adj_index+1, noun_index+1] = 1
                out[noun_index+1, adj_index+1] = 1

    # Ensure no row is all zeros, assign attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the out matrix
    out = out + 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Adjective-Noun Association Pattern", out