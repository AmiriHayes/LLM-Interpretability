from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spaCy model for English
en_nlp = spacy.load('en_core_web_sm')

def noun_adjective_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = en_nlp(sentence)

    # Create a dictionary to align token indices between spaCy and the tokenizer
    tok_aligned = {i: token.idx for i, token in enumerate(doc)}

    for token in doc:
        # Look for 'amod' dependency types, which indicate an adjective modifying a noun
        if token.dep_ == 'amod':
            noun_index = token.head.i
            adj_index = token.i
            # Adjust indices if needed to match tokenizers (this may vary based on actual tokenizer)
            if noun_index in tok_aligned and adj_index in tok_aligned:
                noun_tok_idx = list(tok_aligned.keys())[list(tok_aligned.values()).index(noun_index)]
                adj_tok_idx = list(tok_aligned.keys())[list(tok_aligned.values()).index(adj_index)]
                out[noun_tok_idx, adj_tok_idx] = 1
                out[adj_tok_idx, noun_tok_idx] = 1

    # Attention to end token to ensure there are no rows of zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize by row

    return "Noun-Adjective Association Pattern", out