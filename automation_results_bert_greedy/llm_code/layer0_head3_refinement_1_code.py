import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define the function to generate the attention pattern for coreference and named entity highlighting
def coreference_entity_highlighting(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # defining utility function to help identify similar tokens since we need access to token.text
def spans(sentence: str) -> list:
        words = []
        c_word = ''
        for ch in sentence:
            if ch.isspace():
                if c_word:
                    words.append(c_word)
                    c_word = ''
            else:
                c_word += ch
        if c_word:
            words.append(c_word)
        return words

    sentence_tokens = spans(sentence)
    entity_positions = dict()

    # Identify proper nouns and coreferential elements
    for i, token in enumerate(sentence_tokens):
        if token[0].isupper() and token.lower() not in {"the", "a", "an"}:  # Basic heuristic for entities
            entity_positions[token] = i + 1
        elif token in {"he", "she", "it", "they", "them", "his", "her", "their", "its"}:  # Coreference pronouns
            entity_positions[token] = i + 1

    for entity in entity_positions:
        index = entity_positions[entity]

        if index < len_seq:
            out[index, index] = 1.0  # Self-attention

            if entity in {"he", "she", "it", "they", "them", "his", "her", "their", "its"}:
                for ref_entity, ref_index in entity_positions.items():
                    if ref_entity != entity and ref_index < len_seq:
                        out[index, ref_index] = 1.0
                        out[ref_index, index] = 1.0  # Mutual attention

    # Ensure entries have some non-zero values
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Last token for CLS attention

    # Normalize the output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Coreference and Named Entity Highlighting", out