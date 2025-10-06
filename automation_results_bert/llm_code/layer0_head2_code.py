import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Load the spaCy model
en_core = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization using spaCy for coreference resolution
    doc = en_core(sentence)
    word_to_token_map = {}
    idx = 1
    for token in doc:
        word_to_token_map[token.text.lower()] = idx
        idx += 1

    # Coreference using entity recognition
    for token in doc:
        noun_chunk = token.text.lower()
        if noun_chunk in word_to_token_map:
            token_index = word_to_token_map[noun_chunk]
            for ent in doc.ents:
                if ent.text.lower() in word_to_token_map:
                    ent_index = word_to_token_map[ent.text.lower()]
                    out[token_index, ent_index] = 1
                    out[ent_index, token_index] = 1

    # Ensure no row is all zeros by adding attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreference Resolution Pattern", out