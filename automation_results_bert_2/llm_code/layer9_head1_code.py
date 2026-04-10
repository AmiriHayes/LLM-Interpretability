import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token alignment dictionary
    token_indices = {i: i for i in range(len_seq)}

    # Tokenize the input sentence with spaCy
    spacy_doc = nlp(sentence)

    # Coreference resolution mapping
    coref_pairs = []
    for token in spacy_doc:
        for child in token.children:
            if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                for co_tok in spacy_doc:
                    if co_tok.text.lower() == 'it' or co_tok.text.lower() == 'this':
                        if co_tok.i != token.i:
                            coref_pairs.append((token.i, co_tok.i))

    # Create a mapping between tokens using these pairs
    for source, target in coref_pairs:
        out[token_indices[source]+1, token_indices[target]+1] = 1
        out[token_indices[target]+1, token_indices[source]+1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return 'Coreference Resolution Pattern', out