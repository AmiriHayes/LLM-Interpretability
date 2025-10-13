import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
from typing import Tuple

# Load spaCy language model
en_nlp = spacy.blank('en')


def pronoun_or_named_entity_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy to identify pronouns/NEs
    doc = en_nlp(sentence)
    tokens = [token.text for token in doc]
    attention_index = 0

    # Identify initial token that is a pronoun or proper noun
    for i, token in enumerate(doc):
        if token.pos_ in {"PRON", "PROPN"}:
            attention_index = i
            break

    # Align spaCy tokenization with the Transformer tokenization
    token_alignment = {i: i+1 for i in range(len(tokens))}  # spaCy token to Transformer token index mapping

    # Applying attention pattern to the identified focus
    for j in range(len_seq):
        if j in token_alignment.values() and j != attention_index+1:  # Avoid focusing on oneself
            out[attention_index+1, j] = 1

    # Ensure no row is all zero by self-attention fallback
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Pronoun or Named Entity Focus", out