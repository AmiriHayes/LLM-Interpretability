from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy model for dependency and POS tagging
nlp = spacy.load('en_core_web_sm')

# Define function to identify subordinating conjunctions and prepositions
sub_conj_and_prep = set(['after', 'although', 'because', 'before', 'if', 'once', 'since', 'so', 'that', 'though', 'until', 'when', 'where', 'whether', 'while', 'as', 'for', 'with', 'in', 'on'])


def subordinating_conjunction_and_preposition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    token_to_id = {tok.text: idx + 1 for idx, tok in enumerate(doc)}

    for i, token in enumerate(doc):
        if token.text.lower() in sub_conj_and_prep:
            token_id = token_to_id[token.text]
            out[0, token_id] = 1  # Attention from CLS token
            for child in token.children:
                child_id = token_to_id.get(child.text)
                if child_id:
                    out[token_id, child_id] = 1
            if token.head != token:  # Also consider the head of the conjunction/preposition
                head_id = token_to_id.get(token.head.text)
                if head_id:
                    out[token_id, head_id] = 1

    # Handle self-attention for CLS and EOS
    out[0, 0] = 1  # CLS attends to itself
    out[-1, -1] = 1  # EOS attends to itself

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to EOS as a fallback

    out = out / out.sum(axis=1, keepdims=True)  # Row normalize
    return "Subordinating Conjunction and Preposition Attention", out
