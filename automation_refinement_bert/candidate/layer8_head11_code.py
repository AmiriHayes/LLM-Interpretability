import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

# Function implementing the semantic role focus pattern

def semantic_role_focus_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization and alignment with spaCy
    words = sentence.split()
    doc = nlp(sentence)
    alignment = {token.idx: idx for idx, token in enumerate(doc)}

    # Hypothesis: Focus on main semantic roles (verbs and direct objects)
    for token in doc:
        if token.dep_ in {"ROOT", "dobj", "nsubj"} and token.idx in alignment:
            token_index = alignment[token.idx] + 1 # +1 for [CLS]
            for child in token.children:
                if child.dep_ in {"dobj", "nsubj", "prep", "pobj"} and child.idx in alignment:
                    child_index = alignment[child.idx] + 1
                    out[token_index, child_index] = 1
                    out[child_index, token_index] = 1

    # Ensure no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure it attends to [SEP]

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Semantic Role Focus Pattern", out