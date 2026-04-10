import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
from typing import Tuple

nlp = spacy.load('en_core_web_sm')

def subject_predicate_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    sub_pred_pairs = []

    # Extract subject-predicate pairs using the dependency parser
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            sub_pred_pairs.append((token.i, token.head.i))

    # Map spacy token indices to transformer tokens
    token_map = {}
    word_idx = 0
    for idx, input_id in enumerate(toks.input_ids[0]):
        if tokenizer.decode(input_id).strip():
            token_map[word_idx] = idx
            word_idx += 1

    # Set attention according to subject-predicate pairs
    for sub_idx, pred_idx in sub_pred_pairs:
        if sub_idx in token_map and pred_idx in token_map:
            out[token_map[sub_idx], token_map[pred_idx]] = 1
            out[token_map[pred_idx], token_map[sub_idx]] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Subject-Predicate Emphasis", out

