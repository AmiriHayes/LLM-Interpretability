import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load("en_core_web_sm")

# This function focuses attention primarily on the main object of the sentence or verb.
def object_or_verb_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy to find nouns and verbs
    doc = nlp(sentence)
    noun_or_verb_indices = set()
    for token in doc:
        if token.pos_ in {"NOUN", "VERB"}:
            noun_or_verb_indices.add(token.i)

    # Align spaCy tokens with tokenizer tokens
    word_ids = toks.word_ids()
    for i, word_idx in enumerate(word_ids):
        if word_idx in noun_or_verb_indices:
            # Set high attention to self-attention primarily for object and verb elements
            out[i, i] = 1.0

    # Ensure CLS and SEP have attention
    out[0, 0] = 1.0  # CLS token
    out[-1, -1] = 1.0  # SEP token

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Object or Verb Focused Attention", out