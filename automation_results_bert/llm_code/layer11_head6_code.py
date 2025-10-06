import numpy as np
from transformers import PreTrainedTokenizerBase

# Assumed that spacy and english model are already installed and imported
import spacy
nlp = spacy.load('en_core_web_sm')


def clause_boundary_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Parsing sentence using SpaCy to identify clause boundaries
    doc = nlp(sentence)
    token_dict = {token.idx: token.i for token in doc}  # align token positions

    # Focus attention on likely clause boundaries based on punctuation and token indexes
    for token in doc:
        tok_idx = token.idx
        if token.pos_ in {"PUNCT"} or token.dep_ in {"ROOT", "conj"}:  # clause separators like punctuation or conjunctions
            tokenized_idx = token_dict.get(tok_idx, -1) + 1
            if 1 <= tokenized_idx < len_seq - 1:
                out[tokenized_idx, tokenized_idx] = 1

    # Special attention to [CLS] and [SEP] tokens
    out[0, 0] = 1  # CLS self-attention
    out[-1, -1] = 1  # SEP self-attention

    # Normalize the attention matrix
    out += 1e-4  # Avoid division errors
    out = out / out.sum(axis=1, keepdims=True)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Clause Boundary Detection", out