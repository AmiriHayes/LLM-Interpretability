from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load SpaCy's English tokenizer
nlp = spacy.load('en_core_web_sm')

def initial_noun_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using SpaCy
    doc = nlp(sentence)

    # Find the first noun
    noun_index = None
    for idx, token in enumerate(doc):
        if token.pos_ == 'NOUN':
            noun_index = idx
            break

    # Map SpaCy indices to tokenizer indices
    # There can be more tokens in the tokenized version than in the original sentence;
    # for simplicity, assume alignment is correct if you tokenize the same text once.
    word_to_token = {token.text: idx for idx, token in enumerate(doc)}

    if noun_index is not None:
        tokenized_noun_index = word_to_token.get(doc[noun_index].text, -1)
        if tokenized_noun_index >= 0 and tokenized_noun_index < len_seq:
            out[0, tokenized_noun_index + 1] = 1
            for i in range(1, len_seq):
                out[i, tokenized_noun_index + 1] = 1

    # Normalize
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalization to ensure sum to 1 for each row
    out += 1e-4  # To avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Noun Dominance", out