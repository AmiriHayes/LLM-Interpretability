import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load('en_core_web_sm')


def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = nlp(sentence)

    # Create a mapping from word indices to token ids from tokenizer
    word_to_token_map = {}
    idx = 1
    for token in doc:
        while idx < len_seq and toks.word_ids()[idx] == toks.word_ids()[idx - 1]:
            idx += 1
        word_to_token_map[token.i] = idx
        idx += 1

    # Iterate over named entities and pronouns for potential co-references
    pronoun_tokens = [token for token in doc if token.pos_ == "PRON"]
    noun_tokens = [token for token in doc if token.pos_ in {"NOUN", "PROPN"}]

    for pronoun_token in pronoun_tokens:
        for noun_token in noun_tokens:
            if noun_token.ent_iob_ != 'O' or pronoun_token.text.lower() in {"it", "she", "he", "they"}:
                pronoun_index = word_to_token_map.get(pronoun_token.i, None)
                noun_index = word_to_token_map.get(noun_token.i, None)
                if pronoun_index is not None and noun_index is not None:
                    out[pronoun_index, noun_index] = 1 # Attention from pronoun to noun
                    out[noun_index, pronoun_index] += 1 # Mutual attention

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix row-wise
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreference Resolution Pattern", out