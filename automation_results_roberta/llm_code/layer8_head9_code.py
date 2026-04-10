import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def conjunction_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize with spacy for dependency parsing
    doc = nlp(sentence)  
    # Ensure alignment between tokenizers
    token_alignment = {i: j for i, j in enumerate(toks.word_ids(0)) if j is not None}

    for token in doc:
        if token.dep_ == "cc":  # Coordinating conjunction
            head_idx = token.head.i
            conjunct_idx = token.i
            # Map from spaCy token index to tokenizer token index
            if head_idx in token_alignment and conjunct_idx in token_alignment:
                t_head_idx = list(token_alignment.values()).index(head_idx)
                t_conjunct_idx = list(token_alignment.values()).index(conjunct_idx)

                out[t_head_idx, t_conjunct_idx] = 1
                out[t_conjunct_idx, t_head_idx] = 1

    for row in range(len_seq):  # Guarantee no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Conjunction Dependency Parsing", out