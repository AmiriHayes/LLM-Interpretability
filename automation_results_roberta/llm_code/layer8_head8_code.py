import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")


def noun_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    sentence_spacy = nlp(sentence)

    # Map spaCy token positions to transformer token positions
    spacy_to_transformer = {}
    ti = 0  # Transformer index
    for si, stoken in enumerate(sentence_spacy):
        # Align spaCy tokens to transformer tokens (assumes 1:1 mapping mostly)
        while ti < len(tokens) - 1 and not tokens[ti].startswith(stoken.text):
            ti += 1
        if ti < len(tokens):
            spacy_to_transformer[si] = ti

    # List noun-like POS tags
    noun_tags = {"NOUN", "PROPN", "PRON"}

    # Set attention for nouns and associated tokens
    for token in sentence_spacy:
        if token.pos_ in noun_tags:
            t_index = spacy_to_transformer.get(token.i, None)
            if t_index is not None:
                # Emphasize noun-related token attentions
                for child in token.children:
                    child_t_index = spacy_to_transformer.get(child.i, None)
                    if child_t_index is not None:
                        out[t_index, child_t_index] = 1
                        out[child_t_index, t_index] = 1

    # Ensure CLS and SEP tokens attend to at least themselves
    out[0, 0] = 1  # CLS
    out[len_seq - 1, len_seq - 1] = 1  # SEP

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Noun Focus Pattern", out