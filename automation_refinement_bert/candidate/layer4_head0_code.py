from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def complex_word_formation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    token_to_index = {}
    spacy_tokens = list(doc)

    # Aligning tokens
    for spacy_index, spacy_token in enumerate(spacy_tokens):
        word_id = toks.word_ids(batch_index=0).index(spacy_index)
        token_to_index[word_id] = spacy_index

    for token_id, spacy_index in token_to_index.items():
        spacy_token = spacy_tokens[spacy_index]
        if spacy_token.is_alpha and len(spacy_token) > 3:
            for suffix in spacy_tokens[spacy_index+1:]:
                if suffix.text.startswith('-') or suffix.dep_ in {'compound', 'amod'}:
                    suffix_id = toks.word_ids(batch_index=0).index(suffix.i)
                    out[token_id, suffix_id] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Complex Word Formation Patterns', out

