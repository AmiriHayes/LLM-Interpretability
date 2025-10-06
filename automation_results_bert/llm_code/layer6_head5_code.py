from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")


def word_pair_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_to_token_index = {}
    doc = nlp(sentence)
    token_idx = 0
    for word in doc:
        while token_idx < len(toks.input_ids[0]) - 1 and toks.word_ids()[token_idx] == word.i:
            word_to_token_index[word.i] = token_idx
            token_idx += 1
    def map_dependency(word1_idx, word2_idx):
        if word1_idx in word_to_token_index and word2_idx in word_to_token_index:
            tok_idx1 = word_to_token_index[word1_idx]
            tok_idx2 = word_to_token_index[word2_idx]
            out[tok_idx1, tok_idx2] = 1

    for token1 in doc:
        for token2 in doc:
            # Establishing connections based on manual data exploration (syntactic or co-occurring dependency)
            if token1.head == token2 or token2.head == token1 or token2.dep_ in {"dobj", "pobj", "iobj"}:
                map_dependency(token1.i, token2.i)
            if token1.dep_ == token2.dep_:
                # Apply some relational patterns including co-reference or associated tasks
                map_dependency(token1.i, token2.i)
            if token1.dep_ == "nsubj" and token2.dep_ in {"dobj", "pobj", "iobj"}:
                map_dependency(token1.i, token2.i)

    # Ensure there is no row entirely zeros by default attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize each row to sum to one
    out += 1e-4  # Stabilize to prevent division by zero
    out /= out.sum(axis=1, keepdims=True)

    return "Word-Pair Relationship Focus Pattern", out