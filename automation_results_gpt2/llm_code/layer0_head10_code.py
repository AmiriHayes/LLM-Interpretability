import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def beginning_of_clause_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(sentence)
    token_to_word_mapping = {i: tok.i for i, tok in enumerate(doc)}

    # Map from token index (tokenizer) to word index (spacy)
    word_to_token_mapping = {}
    j = 1  # Skip CLS token
    for word_idx, word in enumerate(doc):
        while j < len(toks.input_ids[0]) and toks.input_ids[0, j] == toks.input_ids[0, j-1]:
            j += 1
        word_to_token_mapping[word_idx] = j
        j += 1

    # We want to give attention to the beginning of each clause
    # Beginning is marked often by subordinator, comma, or first verb
    for word_index, word in enumerate(doc):
        if doc[word_index].pos_ in ["VERB", "AUX", "SCONJ", "CCONJ", "PUNCT"] and (word_index == 0 or doc[word_index-1].pos_ in ["PUNCT", "CCONJ", "SCONJ"]):
            if word_index in word_to_token_mapping:
                token_index = word_to_token_mapping[word_index]
                out[token_index, token_index] = 1

    # Ensure some base self-attention if nothing was found
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to prevent division by zero
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Beginning of Clause Emphasis", out

