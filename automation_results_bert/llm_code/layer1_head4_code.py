import numpy as np
from typing import Tuple
import spacy
from transformers import PreTrainedTokenizerBase

# Load spacy English model
nlp = spacy.load('en_core_web_sm')

def handle_conjunctions(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    mapping = {}
    token_idx = 1  # Start at 1, assuming [CLS] at position 0
    for i, word_token in enumerate(doc):
        word_piece_len = len(tokenizer.tokenize(word_token.text))
        for _ in range(word_piece_len):
            mapping[token_idx] = i
            token_idx += 1

    for token in doc:
        if token.dep_ == 'cc' and token.head.dep_ in {'conj', 'ROOT'}:
            # Link the conjunction to its coordinating conjunction
            conj_idx = token.i
            head_idx = token.head.i
            out[[k for k, v in mapping.items() if v == head_idx], [k for k, v in mapping.items() if v == conj_idx]] = 1
            out[[k for k, v in mapping.items() if v == conj_idx], [k for k, v in mapping.items() if v == head_idx]] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coordination and Conjunction Handling", out