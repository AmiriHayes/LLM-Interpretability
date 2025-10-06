import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load('en_core_web_sm')

def pronominal_reference_and_repetition(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy for part of speech tagging and dependency parsing
    doc = nlp(sentence)
    token_indices = {token.idx: i+1 for i, token in enumerate(doc)}

    # Identify pronominal references and repetitions
    for i, token in enumerate(doc):
        if token.pos_ in ['PRON', 'NOUN', 'PROPN']:
            # Establish reference circles around pronouns/nouns/proper nouns and repetitions
            token_idx = token_indices.get(token.idx, None)
            if token_idx is not None:
                out[token_idx, token_idx] = 1  # Self-attend to the pronoun/noun

            # Link repeated words across the sentence
            for j, other_token in enumerate(doc[i+1:], start=i+1):
                if token.text == other_token.text:
                    other_token_idx = token_indices.get(other_token.idx, None)
                    if other_token_idx is not None:
                        out[token_idx, other_token_idx] = 1
                        out[other_token_idx, token_idx] = 1

    # Ensure each row has at least one attention, usually directed at [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronominal Reference and Repetition Focus", out

