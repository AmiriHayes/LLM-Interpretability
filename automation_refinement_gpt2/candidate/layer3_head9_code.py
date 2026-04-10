import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Load spaCy for English
nlp = spacy.load("en_core_web_sm")

def dominant_root_word_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spaCy
    doc = nlp(sentence)

    # Find the root of the sentence
    root_token = [token for token in doc if token.dep_ == 'ROOT'][0]
    root_index = root_token.i + 1  # Adjust for CLS token

    # Attention on root and its children
    for token in doc:
        if token.head == root_token or token == root_token:
            out[root_index, token.i + 1] = 1
            out[token.i + 1, root_index] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the out matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize return out

    return "Dominant Root Word Emphasis", out