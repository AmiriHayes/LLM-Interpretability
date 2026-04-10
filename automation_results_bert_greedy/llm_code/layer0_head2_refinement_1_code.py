import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def noun_modifier_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the input sentence with spacy for linguistic analysis
    doc = nlp(sentence)
    spacy_token_indices = {token.idx: i for i, token in enumerate(doc)}

    # Traverse Spacy tokens for noun-modifier patterns
    for token in doc:
        if token.dep_ == "amod" or token.dep_ == "det":  # Adjective modifier or determiner
            head_idx = spacy_token_indices[token.head.idx]
            current_idx = spacy_token_indices[token.idx]

            # Link the modifier to its head noun
            out[current_idx + 1, head_idx + 1] = 1
            out[head_idx + 1, current_idx + 1] = 1

    # Ensure CLS and SEP tokens have self-attention
    out[0, 0] = 1.0  # CLS attention
    out[-1, -1] = 1.0  # SEP attention

    # Normalize rows to ensure no row is all zeros, which provides uniform attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Noun-Modifier Linking', out