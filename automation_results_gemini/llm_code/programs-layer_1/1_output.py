import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")

def noun_phrase_chunking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))
    
    # align tokenizer with spacy
    mapping = {}
    k = 1
    for i, token in enumerate(doc):
        indices = [j for j in range(len_seq) if toks.word_ids()[0][j] == i]
        mapping[token.text] = indices

    for i, token in enumerate(doc):
        if token.pos_ in ["DET", "ADJ"]: # Focus on determiners and adjectives
            token_indices = mapping.get(token.text, [])
            if token_indices:
                for j, other_token in enumerate(doc):
                    if (other_token.dep_ == "ROOT" and other_token.pos_ == "NOUN") or (other_token.head == token and other_token.pos_ == 'NOUN'):
                        other_token_indices = mapping.get(other_token.text, [])
                        if other_token_indices:
                            for token_index in token_indices:
                                for other_token_index in other_token_indices:
                                    out[token_index, other_token_index] = 1
                                    out[other_token_index, token_index] = 1

    out[0, 0] = 1 # cls
    out[-1, 0] = 1 # sep
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True) # normalization
    return "Noun Phrase Chunking", out