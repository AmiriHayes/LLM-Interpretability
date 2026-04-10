from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# The function detects the intensifier-adjective pattern
def intensifier_adjective(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = nlp(sentence)
    token_idx_map = {token.idx: i for i, token in enumerate(doc)}

    # Find intensifier-adjective associations
    for token in doc:
        if token.pos_ == 'ADV' and token.dep_ in {'amod', 'advmod'}:
            for child in token.head.children:
                # Check if the child is an adjective
                if child.pos_ == 'ADJ':
                    # Align their indices from the tokenizer
                    adv_index = toks.word_ids.index(token_idx_map[token.idx])
                    adj_index = toks.word_ids.index(token_idx_map[child.idx])
                    out[adv_index][adj_index] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Intensifier-Adjective Association", out