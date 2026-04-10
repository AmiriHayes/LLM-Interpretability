import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def adverbial_clause_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence using spaCy
    doc = nlp(sentence)
    token_map = {token.i: i for i, token in enumerate(doc)}

    # Scan through tokens and identify adverbial clauses
    for token in doc:
        if token.dep_ == 'advcl':
            head_index = token.head.i
            if head_index in token_map:
                head_pos = token_map[head_index]
                clause_index = token_map[token.i]

                # Associate entire adverbial clause with its head
                out[head_pos, clause_index] = 1
                out[clause_index, head_pos] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Adverbial Clause Association", out