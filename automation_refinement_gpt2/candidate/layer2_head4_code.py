import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Using spaCy for NLP tasks like POS tagging and tokenization
nlp = spacy.load('en_core_web_sm')

def sentence_level_noun_phrase_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spacy to find noun phrases
    doc = nlp(sentence)
    noun_phrase_indices = []

    for np in doc.noun_chunks:
        np_start = np.start + 1
        np_end = np.end + 1
        noun_phrase_indices.append((np_start, np_end))

    # Assigning attention scores based on noun phrase origins
    for start, end in noun_phrase_indices:
        for i in range(start, end):
            out[i, start-1] = 1  # Each token in noun phrase attends to first token of the noun phrase

    # Ensure the first token attends to itself
    out[0, 0] = 1

    # If a row has no attention, assign some attention to EOS for continuity
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence-level Dominance on Noun Phrase Origins", out