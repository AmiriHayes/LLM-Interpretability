import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def pronoun_possessive_tracking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy for part-of-speech and dependency parsing
    doc = nlp(sentence)
    token_index_map = {i: tok.i for i, tok in enumerate(doc)}

    # Reference tracking
    for token in doc:
        if token.pos_ in {'PRON', 'DET'}:
            # Find the head of the pronoun or possessive and create attention
            head_index = token.head.i
            token_index = token.i
            if head_index < len_seq and token_index < len_seq:
                out[token_index + 1, head_index + 1] = 1.0

    # Ensure no row is all zeros by attending to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun and Possessive Reference Tracking", out