from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spaCy English tokenizer and POS tagger
nlp = spacy.load('en_core_web_sm')


def adjective_noun_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = nlp(sentence)

    # Map token indices from spaCy to HuggingFace tokens
    hf_spacy_map = {}
    for i, token in enumerate(doc):
        hf_spacy_map[token.i] = i

    # Identify adjective-noun pairs and increment the attention matrix
    for token in doc:
        if token.pos_ == 'ADJ':
            for child in token.children:
                if child.pos_ == 'NOUN' and child.i in hf_spacy_map:
                    adj_idx = hf_spacy_map[token.i] + 1  # Map to BERT token index
                    noun_idx = hf_spacy_map[child.i] + 1
                    out[adj_idx, noun_idx] = 1

    # Self attention
    for i in range(len_seq):
        out[i, i] = 1
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix by row
    out = out / out.sum(axis=1, keepdims=True)

    return "Adjective-Noun Linking Pattern", out

