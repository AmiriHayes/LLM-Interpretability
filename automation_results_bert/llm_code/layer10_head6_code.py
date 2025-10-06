from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')
from typing import Tuple

def focus_on_end_and_objects(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get spaCy tokenization for object identification
    doc = nlp(sentence)
    token_index_map = {token.idx: i+1 for i, token in enumerate(doc)}

    # Focus on objects and ending punctuation
    for token in doc:
        if token.dep_ in {"dobj", "pobj", "nsubj"} or token.pos_ == "NOUN":  # Direct object, prepositional object or noun
            tok_pos = token_index_map[token.idx]
            out[tok_pos, tok_pos] = 0.5  # Moderate self-attention

        # Also focusing ending sentence token (often is a punctuation)
        if token == doc[-1]:
            tok_pos = token_index_map[token.idx]
            out[tok_pos, tok_pos] = 1.0  # Strong self-attention

    # Normalize to avoid zero-summing rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Focus on Sentence Ending and Notable Objects", out